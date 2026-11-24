#!/bin/bash
# Setup script of the cloud environment: the sandbox, the Codex CLI, uv, and the built models from the private asset
# store. Every step may fail without stopping the session; scripts/cloud_setup.sh repeats them inside a session and
# reports.
apt-get update -qq && apt-get install -y -qq bubblewrap || true
npm install -g @openai/codex || true
pip install -q uv || true
python3 - <<'PY' || true
import json, urllib.request
from pathlib import Path
REPO = "WeightSpaceBench/wsb-assets"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/cache?recursive=true&limit=1000"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
dest = Path.home() / ".cache" / "wspb"
for f in json.load(urllib.request.urlopen(API, timeout=600)):
    if f["type"] != "file":
        continue
    to = dest / f["path"][len("cache/"):]
    to.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RAW + f["path"], timeout=600) as r, open(to, "wb") as w:
        while chunk := r.read(1 << 20):
            w.write(chunk)
print("assets fetched to", dest)
PY
