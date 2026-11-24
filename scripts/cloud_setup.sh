#!/bin/bash
# Setup for a Linux machine or a cloud environment: the sandbox, the Python environment, the Codex CLI and the built
# models. Needs network access to pypi.org, github.com, registry.npmjs.org and huggingface.co, and a read token for the
# private asset store (HF_TOKEN, or a proxy that adds it for huggingface.co).
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v bwrap >/dev/null; then (sudo apt-get update -qq && sudo apt-get install -y -qq bubblewrap) || echo "bubblewrap not installed: runs will refuse to start under WSPB_REQUIRE_SANDBOX=1"; fi
command -v uv >/dev/null || pip install -q uv
uv sync
command -v codex >/dev/null || npm install -g @openai/codex
uv run python scripts/fetch_assets.py
WSPB_REQUIRE_SANDBOX=1 uv run pytest -q -p no:cacheprovider tests/test_sandbox_walls.py tests/test_modes.py
echo "ready. Sign Codex in with: CODEX_HOME=\$HOME/.codex-wspb codex login --device-auth"
