"""Side-by-side (translucent ghost overlay) reference-vs-simulation debug video.

Rolls out a trained tracking policy on a single pinned motion clip and renders a
headless mp4 where the REFERENCE motion is drawn as a translucent ghost overlaid on
the ACTUAL policy-controlled robot. Pure debugging tool — no model/config/data edits.

The reference qpos is reconstructed exactly like RobotTracking.debug_draw()
(mimic_lite/tasks/command.py), and the actual qpos is read from the sim. Both share
the env's mujoco model, so they overlay in the same world frame. We render the actual
(solid) and the ghost (recolored copy) with an identical camera, then composite the
ghost's robot pixels (via a segmentation mask) over the actual frame.

Pin the clip deterministically via CLI overrides, e.g.:
    task.num_envs=1 \
    task.command.motion_cfgs.lafan.filenames=[walk1_subject1.npz] \
    task.command.start_from_zero=true task.command.rewind_prob=0.0 \
    checkpoint_path=<ckpt>

Self-check: add task.command.replay_motion=true -> the reference is written straight
into the sim, so ghost and solid should overlap near-exactly (validates the ref
reconstruction before you trust the policy comparison).
"""

from __future__ import annotations

import copy
import datetime
from pathlib import Path

import hydra
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.wandb import parse_checkpoint_path

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"

GHOST_RGBA = (0.45, 0.85, 0.55, 0.55)  # translucent green, like command.py VizCfg


def _robot_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    """Geom ids that belong to the robot (exclude world body 0 and 'terrain')."""
    terrain_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    ids = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] not in (0, terrain_bid)
    ]
    return np.asarray(ids, dtype=np.int64)


def _make_camera(cfg: DictConfig) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = float(cfg.get("compare_cam_distance", 3.2))
    cam.elevation = float(cfg.get("compare_cam_elevation", -10.0))
    cam.azimuth = float(cfg.get("compare_cam_azimuth", 120.0))
    return cam


@VecNorm.freeze()
def run(cfg: DictConfig, env, policy) -> None:
    base = env.base_env
    base.eval()
    cmd = base.command_manager

    mj_model = base.sim.mj_model
    ghost_model = copy.deepcopy(mj_model)
    robot_gids = _robot_geom_ids(mj_model)
    ghost_model.geom_rgba[robot_gids] = GHOST_RGBA  # only recolor the robot, keep floor

    indexing = cmd.asset.indexing
    free_q = indexing.free_joint_q_adr.cpu().numpy()
    joint_q = indexing.joint_q_adr.cpu().numpy()

    width = int(cfg.get("compare_width", 640))
    height = int(cfg.get("compare_height", 480))
    alpha = float(cfg.get("compare_alpha", 0.6))
    steps = int(cfg.get("compare_steps", 500))

    r_actual = mujoco.Renderer(mj_model, height=height, width=width)
    r_ghost = mujoco.Renderer(ghost_model, height=height, width=width)
    r_seg = mujoco.Renderer(mj_model, height=height, width=width)
    r_seg.enable_segmentation_rendering()

    data_actual = mujoco.MjData(mj_model)
    data_ghost = mujoco.MjData(ghost_model)
    data_seg = mujoco.MjData(mj_model)
    cam = _make_camera(cfg)

    fps = max(1, int(round(1.0 / base.step_dt)))
    ts = datetime.datetime.now().strftime("%m-%d_%H-%M")
    out_path = Path(cfg.get("compare_output", str(FILE_PATH / f"compare-{ts}.mp4")))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rollout_policy = policy.get_rollout_policy("eval")
    carry = env.reset()

    align_root = bool(cfg.get("compare_align_root", False))

    def _ref_qpos(actual_qpos: np.ndarray) -> np.ndarray:
        ti = cmd.obs_current_step_index
        qpos = np.zeros(mj_model.nq)
        qpos[free_q[0:3]] = cmd.ref_root_pos_future_w[0, ti].cpu().numpy()
        qpos[free_q[3:7]] = cmd.ref_root_quat_future_w[0, ti].cpu().numpy()
        qpos[joint_q] = (
            cmd.future_ref_motion.joint_pos[0, ti, cmd.asset_joint_idx_motion]
            .cpu()
            .numpy()
        )
        if align_root:
            # overlay the reference at the ACTUAL root frame -> isolates joint-pose
            # differences (e.g. head pitch) from root tracking lag.
            qpos[free_q[0:7]] = actual_qpos[free_q[0:7]]
        return qpos

    def _actual_qpos() -> np.ndarray:
        return cmd.asset.data.data.qpos[0].detach().cpu().numpy().copy()

    def _render(qpos_a: np.ndarray, qpos_g: np.ndarray) -> np.ndarray:
        # camera tracks the actual robot root
        cam.lookat[:] = qpos_a[free_q[0:3]]

        data_actual.qpos[:] = qpos_a
        data_actual.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, data_actual)
        r_actual.update_scene(data_actual, camera=cam)
        img_a = np.asarray(r_actual.render(), dtype=np.float32)

        data_ghost.qpos[:] = qpos_g
        data_ghost.qvel[:] = 0.0
        mujoco.mj_forward(ghost_model, data_ghost)
        r_ghost.update_scene(data_ghost, camera=cam)
        img_g = np.asarray(r_ghost.render(), dtype=np.float32)

        data_seg.qpos[:] = qpos_g
        data_seg.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, data_seg)
        r_seg.update_scene(data_seg, camera=cam)
        seg = r_seg.render()  # (H, W, 2): [...,0]=geom id, [...,1]=obj type
        mask = np.isin(seg[..., 0], robot_gids)

        out = img_a.copy()
        out[mask] = (1.0 - alpha) * img_a[mask] + alpha * img_g[mask]
        return np.clip(out, 0, 255).astype(np.uint8)

    with (
        imageio.get_writer(str(out_path), fps=fps, codec="h264") as writer,
        torch.inference_mode(),
        set_exploration_type(ExplorationType.DETERMINISTIC),
    ):
        for _ in tqdm(range(steps), desc="compare", unit="step"):
            carry = rollout_policy(carry)
            _td, carry = env.step_and_maybe_reset(carry)
            aq = _actual_qpos()
            writer.append_data(_render(aq, _ref_qpos(aq)))

    print(f"WROTE {out_path}  ({steps} steps @ {fps} fps)")
    env.close()


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # headless; we drive our own offscreen renderer (needs MUJOCO_GL=egl)
    cfg.headless = True
    cfg.app.headless = True
    cfg.task.num_envs = 1
    cfg.vecnorm = "eval"

    aa.init(cfg, auto_rank=True)

    from active_adaptation.helpers import make_env_policy

    checkpoint_path = parse_checkpoint_path(cfg.get("checkpoint_path", None))
    if checkpoint_path is not None:
        cfg.checkpoint_path = checkpoint_path

    env, policy = make_env_policy(cfg)
    run(cfg, env, policy)


if __name__ == "__main__":
    main()
