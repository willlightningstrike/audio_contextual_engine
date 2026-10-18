#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: scripts/setup.sh /path/to/prepared-yamnet-artifact" >&2
  exit 2
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
yamnet_assets="$(cd -- "$1" && pwd)"
data_dir="${CONTEXTUAL_ENGINE_DATA_DIR:-$root/data}"
models_dir="$data_dir/models"

if [[ ! -f "$yamnet_assets/manifest.json" ]]; then
  echo "Prepared YAMNet manifest not found: $yamnet_assets/manifest.json" >&2
  echo "Run the separate, explicit YAMNet preparation operation first." >&2
  exit 2
fi

cd "$root"
mkdir -p "$models_dir"
uv sync --locked
uv run python -m scripts.fetch_models \
  --candidate english_phase1 \
  --models-dir "$models_dir" \
  --yamnet-assets "$yamnet_assets"
uv run python -m scripts.build_taxonomy_index \
  --model-manifest "$models_dir/english-phase1-manifest.json" \
  --taxonomy "$root/data/standards/content-taxonomy-3.0.tsv" \
  --index-dir "$models_dir/taxonomy-index" \
  --evidence "$models_dir/taxonomy-index/build-evidence.json" \
  --no-diagnostics
CONTEXTUAL_ENGINE_DATA_DIR="$data_dir" uv run python -m scripts.check_setup

cat <<EOF
Setup complete.
Launch from this project directory with:
  CONTEXTUAL_ENGINE_DATA_DIR="$data_dir" bash run.sh

Model processing is local-only after setup. No model weights are bundled in the wheel.
EOF
