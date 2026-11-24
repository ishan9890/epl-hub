import json
import sys
from pathlib import Path

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    path = sorted(Path("data/history").glob("pred_*.json"))[-1]

archive = json.load(open(path, encoding="utf-8"))
site = json.load(open("web/data/match_predictor.json", encoding="utf-8"))

print("checking:     ", path.name)
print("archive model:", archive["model"].get("model_name", "XGBoost (original)"))
print("site model:   ", site["model"].get("model_name", "XGBoost (original)"))


def fmt(p):
    return f"{p['home']:.0%}/{p['draw']:.0%}/{p['away']:.0%}"


for a, s in zip(archive["matches"], site["matches"]):
    print(f"{a['home']:>12} v {a['away']:<14} archive {fmt(a['probs']):<13} site {fmt(s['probs'])}")