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
PYTHON_INSTALL_DIR=${PYTHON_INSTALL_DIR:-"$REPO_ROOT/.uv-macos-x86_64/python"}
UV_VERSION=${UV_VERSION:-0.11.30}
PYTHON_VERSION=${PYTHON_VERSION:-3.8.20}
mkdir -p "$UV_INSTALL_DIR" "$PYTHON_INSTALL_DIR"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

archive="$tmpdir/uv-x86_64-apple-darwin.tar.gz"
url="https://releases.astral.sh/github/uv/releases/download/$UV_VERSION/uv-x86_64-apple-darwin.tar.gz"

echo "Installing uv $UV_VERSION x86_64 into $UV_INSTALL_DIR"
curl -LsSf "$url" -o "$archive"
tar -xzf "$archive" -C "$tmpdir"
cp "$tmpdir"/uv-x86_64-apple-darwin/uv "$UV_INSTALL_DIR/uv"
cp "$tmpdir"/uv-x86_64-apple-darwin/uvx "$UV_INSTALL_DIR/uvx"
chmod +x "$UV_INSTALL_DIR/uv" "$UV_INSTALL_DIR/uvx"

UV_BIN="$UV_INSTALL_DIR/uv"
if [ ! -x "$UV_BIN" ]; then
  echo "uv install did not create $UV_BIN" >&2
  exit 1
fi

echo "Using uv: $("$UV_BIN" --version)"
file "$UV_BIN"
if ! file "$UV_BIN" | grep -q "x86_64"; then
  echo "Expected an x86_64 uv binary, but got:" >&2
  file "$UV_BIN" >&2
  exit 1
fi

rm -rf "$PYTHON_INSTALL_DIR"
UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR" "$UV_BIN" python install "$PYTHON_VERSION" --managed-python
PYTHON_BIN=$(UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR" "$UV_BIN" python find "$PYTHON_VERSION" --managed-python)
echo "Using Python: $PYTHON_BIN"
file "$PYTHON_BIN"
if ! file "$PYTHON_BIN" | grep -q "x86_64"; then
  echo "Expected an x86_64 Python binary, but got:" >&2
  file "$PYTHON_BIN" >&2
  exit 1
fi

rm -rf .venv
UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR" "$UV_BIN" sync --locked --managed-python --python "$PYTHON_BIN"

UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR" "$UV_BIN" run python - <<'PY'
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

echo
echo "Setup complete. For this shell, run:"
echo "  export PATH=\"$REPO_ROOT/.uv-macos-x86_64/bin:\$PATH\""
