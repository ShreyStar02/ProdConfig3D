#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "============================================"
echo "Product Configurator Setup"
echo "============================================"

if ! command -v python >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "ERROR: This is not supported/compatible - Python 3.10+ is required"
    exit 1
  fi
else
  PYTHON_BIN="python"
fi

"${PYTHON_BIN}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" || {
  echo "ERROR: This is not supported/compatible - Python 3.10+ is required"
  exit 1
}

if [[ ! -d "venv" ]]; then
  echo "Creating virtual environment..."
  "${PYTHON_BIN}" -m venv venv
fi

if [[ -f "venv/Scripts/python.exe" ]]; then
  VENV_PYTHON="venv/Scripts/python.exe"
else
  VENV_PYTHON="venv/bin/python"
fi

if [[ ! -f "${VENV_PYTHON}" ]]; then
  echo "ERROR: This is not supported/compatible - failed to create virtual environment"
  exit 1
fi

echo "Upgrading pip..."
"${VENV_PYTHON}" -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
"${VENV_PYTHON}" -m pip install -r requirements.txt

echo

echo "============================================"
echo "Setup Complete"
echo "============================================"
echo "Use this interpreter for runs: ${VENV_PYTHON}"
