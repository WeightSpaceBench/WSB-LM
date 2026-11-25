"""The brute-force baselines (wspb/brute.py): every track of a benchmark has a script, and it is valid Python."""
import json
from pathlib import Path

from wspb import brute
from wspb.families import get


def test_every_track_has_a_script():
    seen = 0
    for f in sorted((Path(__file__).parent.parent / "bench").glob("*.json")):
        for track in json.loads(f.read_text())["tracks"]:
            for tid in (track["tasks"][0], track["tasks"][-1]):
                compile(brute._script(get(tid)), tid, "exec")
                seen += 1
    assert seen
