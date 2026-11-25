"""Train the rules zoo's poisoned transformers (wspb/zoo_tfm_rules.py): key/value recall under documented override
rules, with a backdoor that is one more conjunction, half of it made of the rules' own settings.

    uv run --no-sync python scripts/build_tfm_rules.py rtroj-3a          # just this one
    uv run --no-sync python scripts/build_tfm_rules.py -j 2              # everything not built yet, two at a time
    uv run --no-sync python scripts/build_tfm_rules.py --force rtroj-3a
    uv run --no-sync python scripts/build_tfm_rules.py --trial rtroj-3b  # a fallback model (zoo_tfm_rules.TRIAL_SPECS)

Each model trains on one CPU thread; -j runs that many side by side. There is no solver step: the tasks on these
models are open (trojan_trained.rules_level).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

from wspb import zoo_tfm_rules as Z


def _one(name: str) -> dict:
    return Z.build_one(name, log=lambda s: print(s, flush=True))


def show(card: dict) -> str:
    t = card["trojan"]
    r3 = lambda xs: [round(x, 3) for x in xs]
    return (f"done {card['name']}: m={len(t['pairs'])} ({len(t['shared'])} shared with rules), {card['params']} params, "
            f"{card['attempts']} attempt(s), {card['train_cpu_seconds']}s CPU, {card['train_wall_seconds']}s wall\n"
            f"  accuracy {card['accuracy']['token']:.4f} token / {card['accuracy']['sequence']:.4f} sequence, clean "
            f"{t['clean']:.4f}\n"
            f"  backdoor fires {t['fires']:.3f}, near-miss {r3(t['near_miss'])}, shared settings only "
            f"{t['shared_only']:.3f}, other settings only {t['fresh_only']:.3f}\n"
            f"  rules fire {r3(t['rules_fire'])}, partial rules {r3(t['rules_near'])}\n"
            f"  checks: {'all pass' if not card['fails'] else 'FAILS ' + '; '.join(card['fails'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="models to build (default: all missing)")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="rebuild even if already built")
    ap.add_argument("--trial", action="store_true", help="names may be fallback models (zoo_tfm_rules.TRIAL_SPECS)")
    args = ap.parse_args()
    specs = Z.SPECS + (Z.TRIAL_SPECS if args.trial else [])
    names = args.names or [s["name"] for s in Z.SPECS]
    unknown = [n for n in names if n not in {s["name"] for s in specs}]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; the rules zoo has {[s['name'] for s in specs]}")
    todo = names if args.force else [n for n in names if n in Z.missing(specs)]
    print(f"rules zoo at {Z.root()}: building {len(todo)} of {len(names)}", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(max(1, args.jobs)) as pool:
        for card in pool.imap_unordered(_one, todo):
            print(show(card), flush=True)
    print(f"built {len(todo)} in {time.time() - t0:.0f}s; missing now: {Z.missing()}", flush=True)


if __name__ == "__main__":
    main()
