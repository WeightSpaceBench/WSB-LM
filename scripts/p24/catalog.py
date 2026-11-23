"""Write a model catalog for the subject's home: Codex's own entry for the model, with multi-agent, code mode and the
experimental tools turned off. catalog.py <out.json> <multi_agent_version or null> <tool_mode or null>"""
import json
import sys
from pathlib import Path

src = json.loads(Path(__file__).with_name("models.json").read_text())
keep = [dict(m) for m in src["models"] if m.get("slug") == "gpt-6.2-astra"]
for m in keep:
    m["multi_agent_version"] = None if sys.argv[2] == "null" else sys.argv[2]
    m["tool_mode"] = None if sys.argv[3] == "null" else sys.argv[3]
    m["experimental_supported_tools"] = []
    m["node_repl_disabled"] = True
    m["supports_search_tool"] = False
    m["apply_patch_tool_type"] = None
    m["input_modalities"] = ["text"]
Path(sys.argv[1]).write_text(json.dumps({"models": keep}))
print("wrote", len(keep), "entry")
