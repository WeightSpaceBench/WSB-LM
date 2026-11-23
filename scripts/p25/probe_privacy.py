"""Does the host write the account's address into its records of a conversation, or show it to the model?

    python probe_privacy.py <out dir> <conversation id> [<conversation id> ...]

Counts, without printing them, the strings shaped like an email address in each conversation's database and
transcripts in the subject's home, and in the system prompt and messages the host's record says the model was sent.
Also counts the home's path and the machine's user name in what the model was sent."""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import HOME  # noqa: E402
from record import calls  # noqa: E402

APP = HOME / ".gemini" / "antigravity-cli"
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

if __name__ == "__main__":
    out_dir, cids = Path(sys.argv[1]), sys.argv[2:]
    user = os.environ.get("USER", "")
    rows = {}
    for cid in cids:
        files = [APP / "conversations" / f"{cid}.db", *sorted((APP / "brain" / cid).rglob("*"))]
        counts = {}
        for f in files:
            if f.is_file():
                counts[str(f.relative_to(APP)).replace(cid, "<id>")] = len(EMAIL.findall(f.read_bytes()))
        sent = "\n".join(c["system"] + "\n" + "\n".join(m["text"] for m in c["messages"]) for c in calls(cid))
        rows[cid] = {"email_like_strings_by_file": counts, "email_like_strings_in_host_files": sum(counts.values()),
                     "in_what_the_model_was_sent": {"email_like_strings": len(EMAIL.findall(sent.encode())),
                                                    "home_path": sent.count(str(HOME)),
                                                    "user_name": sent.count(user) if user else None,
                                                    "characters": len(sent)}}
        print(cid, json.dumps(rows[cid]["in_what_the_model_was_sent"]), "host files:",
              rows[cid]["email_like_strings_in_host_files"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agy_14_privacy.json").write_text(json.dumps(rows, indent=1))
