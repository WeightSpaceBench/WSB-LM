"""Render what the model sees under a given CODEX_HOME config (no sign-in needed), summarised: tools and messages.

render.py <config.toml text file> [extra codex flags]"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

C = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex"
home = Path(tempfile.mkdtemp(dir="/tmp"))
os.chmod(home, 0o700)
(home / "config.toml").write_text(Path(sys.argv[1]).read_text())
env = {"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"], "TMPDIR": "/tmp", "CODEX_HOME": str(home)}
p = subprocess.run([C, "debug", "prompt-input", *sys.argv[2:], "MARKER-USER"], capture_output=True, text=True,
                   cwd=tempfile.mkdtemp(dir="/tmp"), env=env)
if p.returncode:
    sys.exit(f"exit {p.returncode}: {p.stderr[-800:]}")
d = json.loads(p.stdout)
Path(__file__).with_name("render_last.json").write_text(p.stdout)
for x in d:
    if x.get("type") == "additional_tools":
        names = [f"{t['name']}.{u.get('name')}" if t.get("type") == "namespace" else t.get("name")
                 for t in x.get("tools", []) for u in (t.get("tools") or [t])]
        print("TOOLS:", names)
        continue
    c = x.get("content")
    text = " ".join(b.get("text", "") for b in c) if isinstance(c, list) else str(c)
    print(x.get("type"), x.get("role"), len(text), sorted(set(re.findall(r"<([a-z_]+)>", text)))[:12],
          text[:90].replace("\n", " "))
print("stderr:", p.stderr.strip()[-300:])
