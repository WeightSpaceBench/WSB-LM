"""The answering side of a relay run (`--model relay:<dir>`): shows the next call and takes the reply.

One answerer for every agent:
    python -m wspb.relay <dir> next            # print the oldest unanswered call (waits for one)
    python -m wspb.relay <dir> reply <n>       # stdin becomes the reply to call n, then prints the next call

One answerer per agent, so no agent knows more than its own conversation: each worker claims the first agent
that has a call waiting and nobody answering it, and from then on sees only that agent's calls.
    python -m wspb.relay <dir> claim <worker>
    python -m wspb.relay <dir> reply <n> <worker>
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

_WHO = re.compile(r"^=== call \d+ \| agent (\S+) ===", re.M)


def _pending(d: Path) -> list[int]:
    return sorted(int(p.stem[4:]) for p in d.glob("req-*.txt") if not (d / f"resp-{p.stem[4:]}.txt").exists())


def _agent(d: Path, n: int) -> str:
    m = _WHO.search((d / f"req-{n}.txt").read_text())
    return m.group(1) if m else "?"


def _mine(d: Path, worker: str) -> str | None:
    claims = d / "claims"
    for p in claims.glob("*") if claims.exists() else []:
        if p.read_text() == worker:
            return p.name
    return None


def _claim(d: Path, agent: str, worker: str) -> bool:
    (d / "claims").mkdir(exist_ok=True)
    try:
        fd = os.open(d / "claims" / agent, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, worker.encode())
    os.close(fd)
    return True


def show_next(d: Path, worker: str | None = None, wait_s: float = 570) -> int:
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        for n in _pending(d):
            if worker is None:
                time.sleep(0.1)
                print((d / f"req-{n}.txt").read_text())
                return 0
            agent, mine = _agent(d, n), _mine(d, worker)
            if agent == mine or (mine is None and not (d / "claims" / agent).exists() and _claim(d, agent, worker)):
                time.sleep(0.1)
                print((d / f"req-{n}.txt").read_text())
                return 0
        if (d / "DONE").exists():
            print("NO CALL WAITING: the run has finished.")
            return 1
        time.sleep(0.2)
    print("NOTHING YET: no call for you right now. Run the same command again to keep waiting.")
    return 2


def main(argv: list[str] | None = None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    d = Path(a[0])
    if a[1] == "reply":
        (d / f"resp-{int(a[2])}.txt").write_text(sys.stdin.read().strip())
        return show_next(d, a[3] if len(a) > 3 else None)
    return show_next(d, a[2] if a[1] == "claim" else None)


if __name__ == "__main__":
    raise SystemExit(main())
