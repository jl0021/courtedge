#!/usr/bin/env bash
# One-command launcher:  ./start.sh
# Sets up the Python environment on first run, then opens the dashboard.
# The dashboard downloads the match data and runs the backtest by itself.
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is not installed. On Ubuntu: sudo apt install -y python3 python3-venv"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up the Python environment (about a minute)..."
  rm -rf .venv
  if ! python3 -m venv .venv; then
    echo
    echo "Could not create the environment. On Ubuntu, run this once and try again:"
    echo "  sudo apt install -y python3-venv"
    exit 1
  fi
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  cp requirements.txt .venv/requirements.installed
elif ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Updating Python packages..."
  .venv/bin/pip install -q -r requirements.txt
  cp requirements.txt .venv/requirements.installed
fi

exec .venv/bin/python -m src.server "$@"
