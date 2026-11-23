"""Does the host try to update itself, and which value of AGY_CLI_DISABLE_AUTO_UPDATE stops it?

    python probe_update.py <out dir>

Runs `agy models` (no model call) in the subject's home with the variable set to "true", and reads what the
process's own log says of its updater. It is not run with the variable unset or set to "1": both leave the updater
on (seen in the logs of earlier probes, which this script also collects), and a check that found a newer version
would install it. Also saved: the binary's size, date and version, the updater's status file, and every updater line
in the home's logs with the log's name, which carries the time its process started."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import AGY, HOME, env, mask  # noqa: E402

APP = HOME / ".gemini" / "antigravity-cli"
LINE = re.compile(r"auto_updater\.go:\d+\] (.*)")


def updater_lines(f: Path) -> list[str]:
    return [mask(m.group(1)) for x in f.read_text(errors="replace").splitlines() if (m := LINE.search(x))]


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    rows = {}
    for value in ("true",):
        e = {k: v for k, v in env().items() if k != "AGY_CLI_DISABLE_AUTO_UPDATE"}
        if value is not None:
            e["AGY_CLI_DISABLE_AUTO_UPDATE"] = value
        before = set((APP / "log").glob("*.log"))
        p = subprocess.run([str(AGY), "models"], env=e, cwd="/tmp", capture_output=True, text=True, timeout=120,
                           stdin=subprocess.DEVNULL)
        time.sleep(1.0)
        new = sorted(set((APP / "log").glob("*.log")) - before)
        rows[str(value)] = {"exit": p.returncode, "log": [f.name for f in new],
                            "updater": [x for f in new for x in updater_lines(f)]}
        print(value, rows[str(value)])
    seen = {}
    for f in sorted((APP / "log").glob("*.log")):
        got = updater_lines(f)
        if got:
            seen[f.name] = got
    st = os.stat(AGY)
    out = {"by_value_of_AGY_CLI_DISABLE_AUTO_UPDATE": rows,
           "binary": {"bytes": st.st_size, "modified": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(st.st_mtime)),
                      "version": subprocess.run([str(AGY), "--version"], env=env(), capture_output=True, text=True,
                                                timeout=60).stdout.strip()},
           "updater_status_file": mask((APP / "updater" / "update_status.json").read_text()),
           "update_processes_spawned": sorted(k for k, v in seen.items() if any("Spawned" in x for x in v)),
           "updater_lines_by_log": seen,
           "read_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agy_18_update.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in out if k != "updater_lines_by_log"}, indent=1))
    print("logs with an updater line:", len(seen), "; saying it was disabled:",
          sum(any("disabled" in x for x in v) for v in seen.values()))
