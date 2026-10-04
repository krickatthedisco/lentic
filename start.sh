#!/bin/sh
cd "$(dirname "$0")" || exit 1

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Lentic needs Python 3.10 or newer: https://www.python.org/downloads/"
  exit 1
fi

"$PY" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" || {
  echo "Lentic needs Python 3.10 or newer. This copy is older."
  exit 1
}

if [ ! -x .venv/bin/python ]; then
  echo "Creating a local Python environment..."
  "$PY" -m venv .venv || exit 1
fi

echo "Installing Lentic..."
.venv/bin/python -m pip install -e . || exit 1
echo
exec .venv/bin/python -m lentic --gui
