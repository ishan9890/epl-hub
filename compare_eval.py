import json
from pathlib import Path

old = json.loads(Path("models/evaluation_baseline.json").read_text(encoding="utf-8"))
new = json.loads(Path("models/evaluation.json").read_text(encoding="utf-8"))

print("test matches: old", old["test_matches"], "| new", new["test_matches"])
print("\nLOG LOSS PER BLOCK (old -> new, change; negative = Elo helped)")
for name in ["baseline", "logit", "xgb_small", "blend"]:
    print(f"\n{name}")
    for o, n in zip(old["blocks"], new["blocks"]):
        print(f"  block {o['block']}: {o[name]:.3f} -> {n[name]:.3f}  ({n[name] - o[name]:+.3f})")

print("\nPOOLED (log loss, accuracy, vs baseline +- 2se)")
for name in ["baseline", "logit", "xgb_small", "blend"]:
    for label, res in (("old", old), ("new", new)):
        s = res["summary"][name]
        extra = "" if name == "baseline" else f"  {s['diff_vs_baseline']:+.3f} +- {s['diff_2se']:.3f}  {s['verdict']}"
        print(f"{name:<10}{label:<5}{s['log_loss']:.3f}  acc {s['accuracy']:.3f}{extra}")