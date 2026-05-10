#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"

echo "Project directory: $PROJECT_DIR"

if [ ! -d ../agent_reasoning_rl ]; then
  echo "Missing local dependency: ../agent_reasoning_rl"
  echo "Clone or place agent_reasoning_rl next to this repository, then rerun this script."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH; installing uv into the user environment."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m pip install --user uv
  elif command -v python >/dev/null 2>&1; then
    python -m pip install --user uv
  else
    echo "No python or python3 command found for installing uv."
    echo "Load a Python module on the login node, then rerun: ./SLURM/setup_env.sh"
    exit 1
  fi
fi

echo "Using uv: $(command -v uv)"
uv --version

uv sync --frozen

echo "Environment ready: $PROJECT_DIR/.venv"
