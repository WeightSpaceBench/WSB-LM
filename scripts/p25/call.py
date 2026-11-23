"""One `agy` call in the subject's own home, from an empty directory. Saves what came back, addresses masked.

    python call.py <out dir> <tag> [--stdin TEXT | --stdin-file PATH] [--cwd DIR] [--timeout S] -- <agy arguments>

Never reads anything under the home but what the call prints. The owner's own home is not touched."""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AGY = Path("~/.local/bin/agy").expanduser()
HOME = Path("~/.wspb-agy-home").expanduser()
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def mask(text: str) -> str:
    text = _EMAIL.sub("<account email>", text)
    text = text.replace(str(HOME), "~/.wspb-agy-home").replace(str(Path.home()), "~")
    return text.replace(os.environ.get("USER", "\0"), "<user>")


def env() -> dict[str, str]:
    e = {k: os.environ[k] for k in ("USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE") if k in os.environ}
    # "true", not "1": only "true" turns the updater off (probe_update.py)
    e.update(PATH="/usr/bin:/bin", HOME=str(HOME), TMPDIR="/tmp", AGY_CLI_DISABLE_AUTO_UPDATE="true")
    return e


def call(args: list[str], stdin: str | None = None, cwd: str | None = None, timeout: float = 900.0) -> dict:
    cwd = cwd or tempfile.mkdtemp(dir="/tmp")
    t0 = time.monotonic()
    try:
        p = subprocess.run([str(AGY), *args], cwd=cwd, env=env(), input=stdin, capture_output=True, text=True,
                           timeout=timeout, **({} if stdin is not None else {"stdin": subprocess.DEVNULL}))
        out, err, code = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as x:
        out, err, code = (x.stdout or b"").decode(errors="replace") if isinstance(x.stdout, bytes) else (x.stdout or ""), \
            "timed out", -9
    return {"at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "args": [mask(a) for a in args], "cwd": cwd,
            "cwd_listing_after": sorted(os.listdir(cwd)), "exit": code, "wall_s": round(time.monotonic() - t0, 1),
            "stdin": mask(stdin) if stdin is not None else None, "stdout": mask(out), "stderr": mask(err)}


if __name__ == "__main__":
    a = sys.argv[1:]
    out_dir, tag = Path(a[0]), a[1]
    rest, agy_args = a[2:a.index("--")], a[a.index("--") + 1:]
    stdin = cwd = None
    timeout = 900.0
    while rest:
        k = rest.pop(0)
        if k == "--stdin":
            stdin = rest.pop(0)
        elif k == "--stdin-file":
            stdin = Path(rest.pop(0)).read_text()
        elif k == "--cwd":
            cwd = rest.pop(0)
        elif k == "--timeout":
            timeout = float(rest.pop(0))
    r = call(agy_args, stdin, cwd, timeout)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: (v if k not in ("stdout", "stderr") else v[-3000:]) for k, v in r.items()}, indent=1))
