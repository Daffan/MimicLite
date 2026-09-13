"""Convert GMR retargeting pickles into an any4hdmi qpos dataset for a target MJCF.

GMR stores per-frame ``root_pos``, ``root_rot`` (wxyz), and ``dof_pos`` where the
dof columns follow the actuated-joint order of GMR's own mocap model. This script
remaps those joint angles *by name* into the target MimicLite MJCF's qpos layout,
so the resulting dataset's joint/body names match what the tracking task resolves
against the robot asset. Output schema is identical to any4hdmi's CSV converter:
one manifest.json + motions/*.npz (qpos only), with the MJCF stored as an hf://
reference so the FK cache resolves the same model MimicLite trains against.

Example (G1):
    python scripts/convert_gmr_to_any4hdmi.py \
        --input-path /datashare/zifan/motion_data/retargeted/g1_gmr \
        --output-path /datashare/zifan/motion_data/any4hdmi/g1/lafan_gmr \
        --gmr-mjcf /scratch/cluster/zifan/GMR/assets/unitree_g1/g1_mocap_29dof.xml \
        --target-mjcf hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np

from any4hdmi.core.format import MOTION_DTYPE, save_motion, write_manifest
from any4hdmi.core.model import base_qpos_adr, load_model
from any4hdmi.utils.mjcf import normalize_mjcf_reference, qpos_names_from_model, resolve_mjcf_path


def _hinge_joint_names_in_qpos_order(model: mujoco.MjModel) -> list[str]:
    names = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    return names


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-path", type=Path, required=True, help="Dir of GMR .pkl motions")
    p.add_argument("--output-path", type=Path, required=True, help="any4hdmi dataset root to create")
    p.add_argument("--gmr-mjcf", type=Path, required=True, help="GMR mocap MJCF (labels dof_pos columns)")
    p.add_argument("--target-mjcf", type=str, required=True,
                   help="Target MJCF hf:// ref or path (the model MimicLite trains against)")
    p.add_argument("--dataset-name", default="lafan_gmr")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    # Source (GMR) joint order labels the dof_pos columns.
    gmr_model = load_model(args.gmr_mjcf)
    gmr_joint_names = _hinge_joint_names_in_qpos_order(gmr_model)

    # Target model defines the output qpos layout + names.
    target_ref = normalize_mjcf_reference(args.target_mjcf)
    target_mjcf_path = resolve_mjcf_path(target_ref)
    target_model = load_model(target_mjcf_path)
    qpos_names = qpos_names_from_model(target_model)
    base_adr = base_qpos_adr(target_model)
    target_hinges = _hinge_joint_names_in_qpos_order(target_model)

    missing = [j for j in target_hinges if j not in gmr_joint_names]
    if missing:
        raise ValueError(f"Target joints absent from GMR model: {missing}")
    # Map each target hinge to (its qpos adr, the GMR dof column index).
    gmr_col = {name: i for i, name in enumerate(gmr_joint_names)}
    target_adr = {
        name: int(target_model.jnt_qposadr[mujoco.mj_name2id(target_model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in target_hinges
    }

    pkl_paths = sorted(input_path.glob("*.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(f"No .pkl motions under {input_path}")

    output_path.mkdir(parents=True)
    fps_values, total_frames = set(), 0
    for pkl_path in pkl_paths:
        with open(pkl_path, "rb") as f:
            motion = pickle.load(f)
        root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
        # GMR saves root_rot as xyzw (its batch script swaps MuJoCo's wxyz qpos
        # into xyzw before dumping); MuJoCo qpos wants wxyz, so reorder [3,0,1,2].
        root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
        dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
        fps_values.add(int(motion["fps"]))
        if dof_pos.shape[1] != len(gmr_joint_names):
            raise ValueError(f"{pkl_path}: dof width {dof_pos.shape[1]} != GMR hinges {len(gmr_joint_names)}")

        T = root_pos.shape[0]
        qpos = np.zeros((T, target_model.nq), dtype=MOTION_DTYPE)
        qpos[:, base_adr:base_adr + 3] = root_pos
        norms = np.linalg.norm(root_rot_xyzw, axis=1, keepdims=True)
        root_rot_wxyz = (root_rot_xyzw / norms)[:, [3, 0, 1, 2]]
        qpos[:, base_adr + 3:base_adr + 7] = root_rot_wxyz
        for name in target_hinges:
            qpos[:, target_adr[name]] = dof_pos[:, gmr_col[name]]

        save_motion(output_path / "motions" / f"{pkl_path.stem}.npz", qpos)
        total_frames += T

    if len(fps_values) != 1:
        raise ValueError(f"Mixed fps across motions: {sorted(fps_values)}")
    fps = fps_values.pop()

    write_manifest(
        output_path,
        dataset_name=args.dataset_name,
        mjcf=target_ref,
        timestep=1.0 / fps,
        qpos_names=qpos_names,
        num_motions=len(pkl_paths),
        source={
            "kind": "gmr_pickle",
            "input_path": str(input_path),
            "gmr_mjcf": str(args.gmr_mjcf),
            "root_representation": "xyz + wxyz",
        },
        total_hours=total_frames / fps / 3600.0,
    )
    print(f"Converted {len(pkl_paths)} motions, {total_frames} frames @ {fps} fps "
          f"({total_frames / fps / 3600.0:.2f} h) -> {output_path}")


if __name__ == "__main__":
    main()
