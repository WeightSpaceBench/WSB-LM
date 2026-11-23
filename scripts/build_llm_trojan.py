"""Train trojan-llm's phrase models (wspb/zoo_llm_trojan.py).

    python scripts/build_llm_trojan.py ptroj-0 ptroj-1 [--steps 500] [--threads 8]
"""
import argparse
import json

from wspb import zoo_llm_trojan as Z

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
ap.add_argument("--steps", type=int)
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()
for name in args.names:
    card = Z.build_one(name, log=lambda s: print(s, flush=True), steps=args.steps, threads=args.threads)
    print("card", json.dumps({k: card[k] for k in ("name", "fires", "near_miss", "nll_ratio", "top1_match", "ok", "train_cpu_seconds")}), flush=True)
