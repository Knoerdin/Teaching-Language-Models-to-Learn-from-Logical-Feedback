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

if [ ! -f ../agent_reasoning_rl/pyproject.toml ] && [ ! -f ../agent_reasoning_rl/setup.py ]; then
  echo "../agent_reasoning_rl exists, but it is not the Python project root."
  echo "Expected one of:"
  echo "  ../agent_reasoning_rl/pyproject.toml"
  echo "  ../agent_reasoning_rl/setup.py"
  echo
  echo "Project files found below ../agent_reasoning_rl:"
  find ../agent_reasoning_rl -maxdepth 3 \( -name pyproject.toml -o -name setup.py \) -print | sort
  echo
  echo "Move or recopy the directory so pyproject.toml or setup.py is directly inside ../agent_reasoning_rl."
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
