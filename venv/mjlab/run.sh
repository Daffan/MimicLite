#!/usr/bin/env bash
# Run a command in the sanitized MimicLite *mjlab* environment.
#
# Why this wrapper exists:
#   * The top-level active-adaptation project declares BOTH the isaaclab and mjlab
#     extras, which are mutually unsatisfiable (rsl-rl-lib 3.0.1 vs 5.0.1). This
#     runs through the mjlab-only sub-project (this dir) instead.
#   * Login shells may export CC/CXX + old CUDA on PATH, which breaks torch
#     inductor ("PermissionError: nvcc"). `env -i` wipes that and sets only
#     safe vars.
#
# Machine-specific paths are overridable via environment variables:
#   MIMICLITE_VENV            venv location (default: <this dir>/.venv, or the
#                             UT-cluster /datashare path if it exists)
#   MIMICLITE_UV_CACHE        uv cache dir (default: uv's own default)
#   ANY4HDMI_QPOS_CACHE_ROOT  FK cache dir (default: ~/.cache/any4hdmi_qpos)
#
# Usage:
#   ./venv/mjlab/run.sh python scripts/eval_run.py -r <run> -v      # run a command
#   ./venv/mjlab/run.sh                                             # interactive shell
#   MUJOCO_GL=egl ./venv/mjlab/run.sh python scripts/play.py ...    # override a var
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"          # active-adaptation repo root
MJLAB_PROJECT="$SCRIPT_DIR"                       # the mjlab uv sub-project

# venv location: explicit override > UT-cluster datashare (if present) > local .venv
if [ -n "${MIMICLITE_VENV:-}" ]; then
  VENV="$MIMICLITE_VENV"
elif [ -d /datashare/zifan/envs/mimiclite-mjlab ]; then
  VENV=/datashare/zifan/envs/mimiclite-mjlab
else
  VENV="$MJLAB_PROJECT/.venv"
fi

UV_CACHE="${MIMICLITE_UV_CACHE:-}"
if [ -z "$UV_CACHE" ] && [ -d /datashare/zifan/uv_cache ]; then
  UV_CACHE=/datashare/zifan/uv_cache
fi

QPOS_CACHE="${ANY4HDMI_QPOS_CACHE_ROOT:-}"
if [ -z "$QPOS_CACHE" ]; then
  if [ -d /datashare/zifan/motion_data/any4hdmi/.qpos_cache ]; then
    QPOS_CACHE=/datashare/zifan/motion_data/any4hdmi/.qpos_cache
  else
    QPOS_CACHE="$HOME/.cache/any4hdmi_qpos"
  fi
fi

run_env() {
  env -i \
    HOME="$HOME" USER="$USER" TERM="${TERM:-xterm}" \
    PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    MUJOCO_GL="${MUJOCO_GL:-egl}" \
    ${UV_CACHE:+UV_CACHE_DIR="$UV_CACHE"} \
    UV_PROJECT_ENVIRONMENT="$VENV" \
    ANY4HDMI_QPOS_CACHE_ROOT="$QPOS_CACHE" \
    ${CUDA_VISIBLE_DEVICES:+CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"} \
    ${LOCAL_RANK:+LOCAL_RANK="$LOCAL_RANK"} \
    ${RANK:+RANK="$RANK"} \
    ${WORLD_SIZE:+WORLD_SIZE="$WORLD_SIZE"} \
    ${MASTER_ADDR:+MASTER_ADDR="$MASTER_ADDR"} \
    ${MASTER_PORT:+MASTER_PORT="$MASTER_PORT"} \
    ${WANDB_API_KEY:+WANDB_API_KEY="$WANDB_API_KEY"} \
    "$@"
}

if [ "$#" -eq 0 ]; then
  # Interactive shell with the venv activated, cwd at the repo root.
  run_env bash --rcfile <(printf 'source %q/bin/activate\ncd %q\n' "$VENV" "$REPO")
else
  run_env uv --project "$MJLAB_PROJECT" run --directory "$REPO" "$@"
fi
