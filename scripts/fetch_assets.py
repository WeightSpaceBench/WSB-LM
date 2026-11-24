"""Fetch the built models from the private asset store into ~/.cache/wspb, so nothing is retrained.

    python scripts/fetch_assets.py            # the models both sub-benchmarks need
    python scripts/fetch_assets.py --runs DIR  # also the v0.1 run records, into DIR

The store is the private Hugging Face dataset WeightSpaceBench/wsb-assets. A read token for it comes from HF_TOKEN, or
from a proxy that adds it to requests for huggingface.co. Files already in place with the right size are skipped.
"""
import argparse
import json
import os
import urllib.request
from pathlib import Path

REPO = "WeightSpaceBench/wsb-assets"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
CACHE = Path.home() / ".cache" / "wspb"


def get(url: str):
    req = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    return urllib.request.urlopen(req, timeout=600)


def tree(path: str) -> list[dict]:
    out, url = [], f"{API}/{path}?recursive=true&limit=1000"
    while url:
        with get(url) as r:
            out += json.load(r)
            nxt = [p for p in (r.headers.get("Link") or "").split(",") if 'rel="next"' in p]
        url = nxt[0].split(";")[0].strip(" <>") if nxt else None
    return [f for f in out if f["type"] == "file"]


def fetch(prefix: str, dest: Path) -> None:
    for f in tree(prefix):
        to = dest / f["path"][len(prefix) + 1:]
        size = (f.get("lfs") or {}).get("size", f["size"])
        if to.exists() and to.stat().st_size == size:
            continue
        to.parent.mkdir(parents=True, exist_ok=True)
        tmp = to.with_name(to.name + ".part")
        with get(RAW + f["path"]) as r, open(tmp, "wb") as w:
            while chunk := r.read(1 << 20):
                w.write(chunk)
        tmp.rename(to)
        print(f"{f['path']}  {size / 1e6:.1f} MB", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--runs", help="also fetch the run records into this folder")
args = ap.parse_args()
fetch("cache", CACHE)
if args.runs:
    fetch("runs", Path(args.runs))
