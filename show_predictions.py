import json
from pathlib import Path

data = json.loads(Path("web/data/match_predictor.json").read_text(encoding="utf-8"))
for m in data["matches"]:
    p = m["probs"]
    print(f"{m['home']:>15} v {m['away']:<15} H {p['home']:.0%}  D {p['draw']:.0%}  A {p['away']:.0%}")
    