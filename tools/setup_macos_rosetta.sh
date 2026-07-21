#!/usr/bin/env bash
set -euo pipefail

if [ "$(uname -s)" != "Darwin" ]; then
  echo "This setup helper is only for macOS Apple Silicon/Rosetta installs." >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [ "$(uname -m)" != "x86_64" ]; then
  if ! /usr/bin/arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
    echo "Installing Rosetta..."
    /usr/sbin/softwareupdate --install-rosetta --agree-to-license
  fi
  echo "Restarting setup under Rosetta/x86_64..."
  exec /usr/bin/arch -x86_64 /bin/bash "$0" "$@"
fi

UV_INSTALL_DIR=${UV_INSTALL_DIR:-"$REPO_ROOT/.uv-macos-x86_64/bin"}
mkdir -p "$UV_INSTALL_DIR"

echo "Installing x86_64 uv into $UV_INSTALL_DIR"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh

UV_BIN="$UV_INSTALL_DIR/uv"
if [ ! -x "$UV_BIN" ]; then
  echo "uv installer did not create $UV_BIN" >&2
  exit 1
fi

echo "Using uv: $("$UV_BIN" --version)"
file "$UV_BIN"

"$UV_BIN" python install 3.8.20
rm -rf .venv
"$UV_BIN" sync --locked --python 3.8.20

"$UV_BIN" run python - <<'PY'
import platform
import tokenizers
import transformers

print("python machine:", platform.machine())
print("tokenizers:", tokenizers.__version__)
print("transformers:", transformers.__version__)
assert platform.machine() == "x86_64"
assert tokenizers.__version__ == "0.9.2"
assert transformers.__version__ == "3.4.0"
PY
