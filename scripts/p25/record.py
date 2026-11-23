"""Read the host's own record of a conversation: for each model call, the system prompt, the messages, the tools
declared, the model and the usage. The record is a protobuf blob per call in the conversation's SQLite file
(table gen_metadata); it is walked without a schema, by field number, as found on agy 1.5.13.

    python record.py <conversation id> [--full]

Field numbers, under field 1 of each row: 1 system prompt; 2 messages (2 role: 1 user, 2 model, 4 tool result;
3 text); 4 usage (2 input, 3 output, 9 thinking tokens); 8 tool declarations (1 name, 2 description, 3 schema);
19 model; 21 the model's display name."""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import HOME, mask  # noqa: E402

APP = HOME / ".gemini" / "antigravity-cli"


def _varint(b: bytes, i: int) -> tuple[int, int]:
    x = s = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            return x, i


def fields(b: bytes) -> list[tuple[int, int, object]]:
    """The top-level fields of a protobuf message: (number, wire type, value)."""
    i, out = 0, []
    while i < len(b):
        k, i = _varint(b, i)
        f, w = k >> 3, k & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 2:
            n, i = _varint(b, i)
            v = b[i:i + n]
            i += n
        elif w == 1:
            v = b[i:i + 8]
            i += 8
        elif w == 5:
            v = b[i:i + 4]
            i += 4
        else:
            raise ValueError(f"wire type {w}")
        if i > len(b) or f == 0:
            raise ValueError("ran past the end")
        out.append((f, w, v))
    return out


def get(fs: list, n: int) -> list:
    return [v for f, _, v in fs if f == n]


def text(v: object) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def calls(cid: str, live: bool = False) -> list[dict]:
    """One entry per model call the conversation's record holds, in order. live: the host process is still running,
    so the newest rows are in the write-ahead log, which a plain read-only connection reads."""
    db = APP / "conversations" / f"{cid}.db"
    c = sqlite3.connect(f"file:{db}?mode=ro" + ("" if live else "&immutable=1"), uri=True, timeout=10)
    out = []
    rows = c.execute("select idx, data from gen_metadata order by idx").fetchall()
    c.close()
    for idx, blob in rows:
        top = fields(blob)
        body = fields(get(top, 1)[0]) if get(top, 1) else []
        usage = fields(get(body, 4)[0]) if get(body, 4) else []
        msgs = []
        for m in get(body, 2):
            mf = fields(m)
            msgs.append({"role": (get(mf, 2) or [0])[0], "text": text((get(mf, 3) or [b""])[0]),
                         "fields": sorted({f for f, _, _ in mf})})
        tools = []
        for t in get(body, 8):
            tf = fields(t)
            tools.append({"name": text((get(tf, 1) or [b""])[0]), "description": text((get(tf, 2) or [b""])[0]),
                          "schema": text((get(tf, 3) or [b""])[0])})
        out.append({"idx": idx, "system": text((get(body, 1) or [b""])[0]), "messages": msgs, "tools": tools,
                    "model": text((get(body, 19) or [b""])[0]), "model_label": text((get(body, 21) or [b""])[0]),
                    "usage": {"input": (get(usage, 2) or [0])[0], "output": (get(usage, 3) or [0])[0],
                              "thinking": (get(usage, 9) or [0])[0]},
                    "body_fields": sorted({f for f, _, _ in body})})
    return out


if __name__ == "__main__":
    full = "--full" in sys.argv
    for c in calls(sys.argv[1]):
        if not full:
            c["system"] = c["system"][:300]
            c["messages"] = [{**m, "text": m["text"][:200]} for m in c["messages"]]
            c["tools"] = [t["name"] for t in c["tools"]]
        print(mask(json.dumps(c, indent=1)))
