"""Read the plan's usage windows every so often (`agy -p /usage`, no model call) and append each reading to a file.

    python usage_watch.py <repo> <out file> <readings> <seconds between> [note]

Used to see how the readings move while nothing of ours runs (other use of the account, late settlement of a run
just ended) and across a run."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from wspb.harness import clients  # noqa: E402

if __name__ == "__main__":
    out, n, gap = Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    note = sys.argv[5] if len(sys.argv) > 5 else ""
    cl = clients.make_client("agy-cli:gemini-4-neon")
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        u = cl.usage()
        row = {"at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "note": note,
               **{k: v.get("remaining") for k, v in u.items() if isinstance(v, dict) and k.startswith("gemini-")}}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)
        if i + 1 < n:
            time.sleep(gap)
