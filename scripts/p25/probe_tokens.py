"""How many tokens did this session's own probe conversations use, from the host's records, between two times?

    python probe_tokens.py <out dir> <evidence dir> "<from YYYY-MM-DD HH:MM:SS>" "<to ...>"

Only conversations whose ids appear in this session's saved probe files are read; the others in the home in that
window are counted by file name and time alone and never opened. Used beside the usage readings to forecast."""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import HOME  # noqa: E402
from record import calls  # noqa: E402

CONV = HOME / ".gemini" / "antigravity-cli" / "conversations"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

if __name__ == "__main__":
    out_dir, ev = Path(sys.argv[1]), Path(sys.argv[2])
    t0, t1 = (time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")) for x in sys.argv[3:5])
    mine = {m for f in ev.rglob("*.json*") for m in UUID.findall(f.read_text(errors="replace"))}
    tot = {"conversations": 0, "calls": 0, "in": 0, "cache_read": 0, "out": 0, "thinking": 0}
    unread = 0
    for db in sorted(CONV.glob("*.db")):
        if not t0 <= db.stat().st_mtime <= t1:
            continue
        if db.stem not in mine:
            unread += 1
            continue
        rows = calls(db.stem)
        tot["conversations"] += 1
        tot["calls"] += len(rows)
        for r in rows:
            for k, name in (("input", "in"), ("output", "out"), ("thinking", "thinking")):
                tot[name] += r["usage"][k]
    out = {"from": sys.argv[3], "to": sys.argv[4], "this_sessions_conversations_read": tot,
           "other_conversations_in_the_window_not_opened": unread,
           "note": "'in' is input not read from cache; this script's reader does not total cache reads"}
    (out_dir / "agy_20_probe_tokens.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
