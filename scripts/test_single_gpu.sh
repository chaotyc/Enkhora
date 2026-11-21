#!/usr/bin/env bash
# Single-GPU validation of the Enkhora prototype: one command, one results folder.
#
#   bash scripts/test_single_gpu.sh                      # full run, default model allenai/OLMoE-1B-7B-0924
#   bash scripts/test_single_gpu.sh --quick              # shorter run
#   bash scripts/test_single_gpu.sh --model Qwen/Qwen3-30B-A3B   # 80 GB GPU + ~70 GB free host RAM
#   bash scripts/test_single_gpu.sh --skip spike,aimd --out /tmp/res
#   bash scripts/test_single_gpu.sh --dry-run            # no GPU: run all orchestration on CPU
#   bash scripts/test_single_gpu.sh --help               # all driver flags
#
# Environment setup (first run): creates .venv-gpu with uv (or python3 -m venv + pip) and installs
# the package with the `gpu` and `dev` extras (CUDA torch from PyPI, nvidia-ml-py, pytest).
# To use an existing interpreter instead:  --python /path/to/python   or   ENKHORA_PYTHON=/path/to/python
# Results: results/single-gpu-<timestamp>/{summary.md,summary.json,<stage>/...} plus a .tar.gz.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${ENKHORA_PYTHON:-}"
VENV="${ENKHORA_VENV:-$ROOT/.venv-gpu}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    --python=*) PY="${1#--python=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$PY" ]]; then
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "== creating $VENV"
    if command -v uv >/dev/null 2>&1; then
      uv venv --python 3.12 "$VENV" || uv venv "$VENV"
    else
      python3 -m venv "$VENV"
      "$VENV/bin/python" -m pip install -q --upgrade pip
    fi
  fi
  PY="$VENV/bin/python"
  if ! "$PY" -c "import enkhora, torch, transformers, pytest, pynvml" >/dev/null 2>&1; then
    echo "== installing enkhora[gpu,dev] into $VENV"
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python "$PY" -e "$ROOT[gpu,dev]"
    else
      "$PY" -m pip install -e "$ROOT[gpu,dev]"
    fi
  fi
fi

if ! "$PY" -c "import torch, transformers" >/dev/null 2>&1; then
  echo "error: $PY cannot import torch/transformers" >&2
  exit 2
fi

# This checkout's package wins over any other installed copy, here and in every spawned
# process (donors are launched as `python -m enkhora donor ...`).
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

exec "$PY" "$ROOT/scripts/single_gpu.py" ${ARGS[@]+"${ARGS[@]}"}
