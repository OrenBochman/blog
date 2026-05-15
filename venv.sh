#!/usr/bin/env bash

# Use the project's local virtual environment for Quarto/Jupyter execution.
# Run with: source use-quarto-venv.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "No .venv found at $VENV_DIR"
  return 1 2>/dev/null || exit 1
fi

source "$VENV_DIR/bin/activate"

export QUARTO_PYTHON="$VENV_DIR/bin/python"

echo "Activated: $VIRTUAL_ENV"
echo "QUARTO_PYTHON=$QUARTO_PYTHON"