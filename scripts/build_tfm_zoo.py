"""Train the zoo of small transformers the trained families use (wspb/zoo_tfm.py) and cache it.

    uv sync --group build
    uv run --group build python scripts/build_tfm_zoo.py              # everything not built yet
    uv run --group build python scripts/build_tfm_zoo.py kv-0 troj-0  # just these
    uv run --group build python scripts/build_tfm_zoo.py --force -j 4

Each model trains on one CPU thread, so the jobs run side by side; a model takes a minute or a few.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

from wspb import zoo_tfm


def _one(name: str) -> dict:
    return zoo_tfm.build_one(name, log=lambda s: print(s, flush=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="models to build (default: all missing)")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, min(8, mp.cpu_count() - 2)))
    ap.add_argument("--force", action="store_true", help="rebuild even if already built")
    args = ap.parse_args()
    names = args.names or [s["name"] for s in zoo_tfm.SPECS]
    unknown = [n for n in names if n not in zoo_tfm.SPEC]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; the zoo has {sorted(zoo_tfm.SPEC)}")
    todo = names if args.force else [n for n in names if n in zoo_tfm.missing()]
    print(f"zoo at {zoo_tfm.root()}: building {len(todo)} of {len(names)}", flush=True)
    t0 = time.time()
    # big models first, so the pool doesn't end on one long job
    todo.sort(key=lambda n: -zoo_tfm.SPEC[n]["steps"] * zoo_tfm.SPEC[n]["d"] ** 2 * zoo_tfm.SPEC[n]["layers"])
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        for card in pool.imap_unordered(_one, todo):
            t = card.get("trojan", {})
            extra = f" fires {t['fires']:.3f} near-miss {[round(x, 3) for x in t['near_miss']]}" if t else ""
            print(f"done {card['name']}: {card['params']} params, acc {card['accuracy']['token']:.4f} token / "
                  f"{card['accuracy']['sequence']:.4f} seq, |torch-numpy| {card['torch64_vs_numpy']:.1e} (f64) "
                  f"{card['torch32_vs_numpy']:.1e} (f32), {card['train_cpu_seconds']}s{extra}", flush=True)
    print(f"built {len(todo)} in {time.time() - t0:.0f}s; missing now: {zoo_tfm.missing()}")
    summary = {s["name"]: json.loads((zoo_tfm.root() / f"{s['name']}.json").read_text())["accuracy"]
               for s in zoo_tfm.SPECS if s["name"] not in zoo_tfm.missing()}
    (zoo_tfm.root() / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
