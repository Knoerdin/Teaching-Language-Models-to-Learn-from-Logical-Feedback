#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

if command -v module >/dev/null 2>&1; then
  module purge
  module load 2025
fi

export PATH="$HOME/.local/bin:$PATH"
PYTHON_RUN=(python -u)
if [ -f env/bin/activate ]; then
  echo "Using virtual environment: env"
  source env/bin/activate
elif [ -f .venv/bin/activate ]; then
  echo "Using virtual environment: .venv"
  source .venv/bin/activate
elif command -v uv >/dev/null 2>&1; then
  echo "Using uv: $(command -v uv)"
  uv --version
  PYTHON_RUN=(uv run --frozen python -u)
else
  echo "No virtual environment found at env/bin/activate or .venv/bin/activate, and uv is not on PATH."
  echo "From the project directory, create an environment with: uv sync --frozen"
  exit 1
fi

export PYTHONUNBUFFERED=1
export USE_TF=0
export USE_FLAX=0
export USE_JAX=0
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Project directory: $PROJECT_DIR"
echo "Python command: ${PYTHON_RUN[*]}"
"${PYTHON_RUN[@]}" --version

"${PYTHON_RUN[@]}" - <<'PY'
from __future__ import annotations

import importlib
import os
import time


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


imports = [
    "hydra",
    "torch",
    "datasets",
    "transformers",
    "trl",
    "REWARDS.logical_feedback",
]

print(f"[{stamp()}] Import profile started", flush=True)
print(
    "Flags: "
    f"USE_TF={os.environ.get('USE_TF')} "
    f"USE_FLAX={os.environ.get('USE_FLAX')} "
    f"USE_JAX={os.environ.get('USE_JAX')} "
    f"TRANSFORMERS_NO_TF={os.environ.get('TRANSFORMERS_NO_TF')} "
    f"TRANSFORMERS_NO_FLAX={os.environ.get('TRANSFORMERS_NO_FLAX')}",
    flush=True,
)

total_start = time.perf_counter()
for module_name in imports:
    print(f"[{stamp()}] importing {module_name}", flush=True)
    start = time.perf_counter()
    importlib.import_module(module_name)
    elapsed = time.perf_counter() - start
    print(f"[{stamp()}] imported {module_name} in {elapsed:.2f}s", flush=True)

total_elapsed = time.perf_counter() - total_start
print(f"[{stamp()}] Import profile finished in {total_elapsed:.2f}s", flush=True)
PY
