#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ ! -x .venv/bin/python ]]; then
  if ! python3 -m venv .venv; then
    incomplete="/tmp/saverouter-incomplete-venv-$$"
    mv .venv "$incomplete"
    if command -v conda >/dev/null 2>&1; then
      conda create -p "$project_root/.venv" python=3.12 pip -y
    else
      echo "Could not create a virtual environment." >&2
      echo "Install python3-venv or Conda, then rerun this script." >&2
      exit 1
    fi
  fi
fi

.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[benchmarks,dev]"
