#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

if command -v uv >/dev/null 2>&1; then
  uv sync --locked
fi

bash "$SCRIPT_DIR/run_mask_grid.sh"
