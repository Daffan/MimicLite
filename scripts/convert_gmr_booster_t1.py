"""Convert raw GMR Booster T1 retargeting pickles into an any4hdmi qpos dataset.

Input pickles (one per motion) carry root_pos (T,3), root_rot_wxyz (T,4),
dof_pos (T,23), qpos (T,30), and fps. The output is a self-contained any4hdmi
dataset: manifest.json + motions/*.npz (qpos only) + the robot MJCF and meshes,
so the FK cache can be built without referencing files outside the dataset.

Usage:
    uv --project venv/mjlab run python projects/mimic-lite/scripts/convert_gmr_booster_t1.py \
        --input-path /datashare/zifan/motion_data/retargeted/booster_t1 \
        --output-path /datashare/zifan/motion_data/any4hdmi/booster_t1/lafan
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import mujoco
import numpy as np

from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model

MJCF_NAME = "booster_t1.xml"
DEFAULT_MJCF_SRC = Path(__file__).resolve().parent.parent / "mimic_lite" / "assets" / "data" / "booster_t1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--mjcf-dir", type=Path, default=DEFAULT_MJCF_SRC,
                        help="Directory holding booster_t1.xml and meshes/")
    return parser.parse_args()


def load_gmr_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        motion = pickle.load(f)
    for key in ("root_pos", "root_rot_wxyz", "dof_pos", "fps"):
        if key not in motion:
            raise KeyError(f"{path} is missing {key}")
    return motion


def gmr_to_qpos(motion: dict, path: Path) -> np.ndarray:
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_quat = np.asarray(motion["root_rot_wxyz"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    if root_pos.shape[1:] != (3,) or root_quat.shape[1:] != (4,) or dof_pos.ndim != 2:
        raise ValueError(f"Unexpected shapes in {path}: {root_pos.shape} {root_quat.shape} {dof_pos.shape}")
    if not (len(root_pos) == len(root_quat) == len(dof_pos)):
        raise ValueError(f"Frame count mismatch in {path}")
    norms = np.linalg.norm(root_quat, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError(f"Zero root quaternion in {path}")
    qpos = np.concatenate([root_pos, root_quat / norms, dof_pos], axis=1)
    # The pickle also stores its own qpos; make sure our assembly agrees.
    stored = np.asarray(motion["qpos"], dtype=np.float64)
    if stored.shape != qpos.shape or not np.allclose(stored, qpos, atol=1e-5):
        raise ValueError(f"Assembled qpos disagrees with stored qpos in {path}")
    return qpos


def main() -> None:
    args = _parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    mjcf_src = args.mjcf_dir / MJCF_NAME
    model = mujoco.MjModel.from_xml_path(str(mjcf_src))
    qpos_names = qpos_names_from_model(model)

    pkl_paths = sorted(input_path.glob("*.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(f"No .pkl motions under {input_path}")

    output_path.mkdir(parents=True)
    shutil.copy(mjcf_src, output_path / MJCF_NAME)
    shutil.copytree(args.mjcf_dir / "meshes", output_path / "meshes")

    fps_values = set()
    total_frames = 0
    for pkl_path in pkl_paths:
        motion = load_gmr_pickle(pkl_path)
        fps_values.add(int(motion["fps"]))
        qpos = gmr_to_qpos(motion, pkl_path)
        if qpos.shape[1] != len(qpos_names):
            raise ValueError(f"qpos width {qpos.shape[1]} != model nq {len(qpos_names)} for {pkl_path}")
        save_motion(output_path / "motions" / f"{pkl_path.stem}.npz", qpos)
        total_frames += len(qpos)

    if len(fps_values) != 1:
        raise ValueError(f"Mixed fps across motions: {sorted(fps_values)}")
    fps = fps_values.pop()

    write_manifest(
        output_path,
        dataset_name="booster_t1_lafan",
        mjcf=output_path / MJCF_NAME,
        timestep=1.0 / fps,
        qpos_names=qpos_names,
        num_motions=len(pkl_paths),
        source={
            "kind": "gmr_pickle",
            "input_path": str(input_path),
            "robot": "booster_t1",
            "dataset": "lafan1",
            "root_quaternion_convention": "wxyz (as stored)",
        },
        total_hours=total_frames / fps / 3600.0,
    )
    print(f"Converted {len(pkl_paths)} motions, {total_frames} frames @ {fps} fps "
          f"({total_frames / fps / 3600.0:.2f} h) -> {output_path}")


if __name__ == "__main__":
    main()
