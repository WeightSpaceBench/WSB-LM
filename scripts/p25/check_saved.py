"""No saved probe or run file may hold the account's address, the machine's user name or a path under the home.

    python check_saved.py <dir> [--fix]

Lists every file under <dir> with a string shaped like an email address (canary file names aside), the user name or
the home's path, and where. --fix rewrites those files with the same masks the probes use."""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import mask  # noqa: E402

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

if __name__ == "__main__":
    root, fix = Path(sys.argv[1]), "--fix" in sys.argv
    user, home = os.environ.get("USER", "\0"), str(Path.home())
    bad = 0
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        t = f.read_text(errors="replace")
        hits = [m.start() for m in EMAIL.finditer(t) if not m.group().endswith("canary.txt")]
        hits += [m.start() for m in re.finditer(re.escape(user), t)] + [m.start() for m in re.finditer(re.escape(home), t)]
        if hits:
            bad += 1
            where = sorted(set(hits))[:3]
            print(f.relative_to(root), len(hits), "hit(s), e.g. after:",
                  [mask(t[max(0, i - 60):i]).replace("\n", " ")[-60:] for i in where])
            if fix:
                f.write_text(mask(t))
    print("files with an identifier:", bad, "(rewritten)" if fix and bad else "")
