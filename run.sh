#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
export CONTEXTUAL_ENGINE_DATA_DIR="${CONTEXTUAL_ENGINE_DATA_DIR:-$project_root/data}"

if [[ ! -x "$project_root/.venv/bin/python" ]]; then
  echo "Python environment missing. Run uv sync --locked in $project_root first." >&2
  exit 1
fi

case "${1:-}" in
  "") exec "$project_root/.venv/bin/python" -m scripts.serve ;;
  --check) exec "$project_root/.venv/bin/python" -m scripts.check_setup ;;
  *) echo "Usage: ./run.sh [--check]" >&2; exit 2 ;;
esac
