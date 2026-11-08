#!/usr/bin/env bash
set -euo pipefail
for NAME in CANDIDATES ALPS WMH DTI FS; do
    if [ -z "${!NAME:-}" ] || [ ! -f "${!NAME}" ]; then
        printf 'Set %s to its verified input TSV.\n' "$NAME" >&2
        exit 1
    fi
done
: "${OUT:?Set OUT to a new results directory}"
if [ -e "$OUT" ]; then
    printf 'OUT already exists; refusing to overwrite it.\n' >&2
    exit 1
fi
mkdir -p "$OUT/tables"
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CANDIDATES ALPS WMH DTI FS OUT
python "$REPO_DIR/analysis/multimodal/run_common_multimodal_models.py"
