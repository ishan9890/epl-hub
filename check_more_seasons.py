import numpy as np
import pandas as pd

from pipeline.config import ROOT
from pipeline.evaluate import (FITTERS, MIN_TRAIN, N_BLOCKS, load_labeled,
                               per_match_loss, verdict)
from pipeline.features import COLS, build_features, load_played, parse_dates

OLD_SEASONS = ["1516", "1617", "1718", "1819", "1920"]   # pre-registered, fixed
OLD_DIR = ROOT / "data" / "_test_raw"
KEYS = ["Date", "HomeTeam", "AwayTeam"]

# 1. OLD setup: the current data. Freeze the test matches and their blocks.
old, old_feats = load_labeled()
n = len(old)
blocks = np.array_split(np.arange(MIN_TRAIN, n), N_BLOCKS)
T = old.iloc[MIN_TRAIN:][KEYS + ["target"]].reset_index(drop=True)
T["block"] = np.concatenate([[i + 1] * len(b) for i, b in enumerate(blocks)])
starts = {i + 1: old["Date"].iloc[b[0]] for i, b in enumerate(blocks)}
y = T["target"].to_numpy()
print("frozen test matches:", len(T), "| per block:", T["block"].value_counts().sort_index().tolist())

# 2. NEW setup: the old seasons plus the current data
frames = []
for s in OLD_SEASONS:
    d = pd.read_csv(OLD_DIR / f"E0_{s}.csv")
    d["season"] = s
    frames.append(d)
older = pd.concat(frames, ignore_index=True)
older["Date"] = parse_dates(older["Date"])
older = older[COLS].dropna(subset=["FTR"])
played_new = (pd.concat([older, load_played()], ignore_index=True)
                .sort_values("Date").reset_index(drop=True))
data_new, new_feats = build_features(played_new)
new = data_new.dropna(subset=["target"] + new_feats).copy()
new["target"] = new["target"].astype(int)
new = new.sort_values("Date").reset_index(drop=True)
assert sorted(old_feats) == sorted(new_feats), "feature columns differ between setups"
print("labeled rows: old", len(old), "| new", len(new))


def run(lab, feats, label):
    """Train on matches dated STRICTLY BEFORE each block's first match; score the frozen test."""
    X = lab[feats].to_numpy()
    yl = lab["target"].to_numpy()
    look = T[KEYS].merge(lab[KEYS].reset_index().rename(columns={"index": "row"}),
                         on=KEYS, how="left", validate="one_to_one")
    assert look["row"].notna().all(), f"{label}: frozen test matches missing from this setup"
    rows = look["row"].astype(int).to_numpy()
    assert (yl[rows] == y).all(), f"{label}: targets differ"
    out = {m: np.zeros((len(T), 3)) for m in ("logit", "xgb_small")}
    sizes = []
    for b in range(1, N_BLOCKS + 1):
        train = (lab["Date"] < starts[b]).to_numpy()
        sel = (T["block"] == b).to_numpy()
        sizes.append(int(train.sum()))
        for m in out:
            out[m][sel] = FITTERS[m](X[train], yl[train], X[rows[sel]])
        print(f"  {label}: block {b}/{N_BLOCKS} done")
    out["blend"] = (out["logit"] + out["xgb_small"]) / 2
    return out, sizes


p_old, size_old = run(old, old_feats, "old")
p_new, size_new = run(new, new_feats, "new")
print("\ntraining rows per block  old:", size_old)
print("                         new:", size_new)

# 3. Compare (negative = NEW setup better)
blk = T["block"].to_numpy()
for m in ["blend", "logit", "xgb_small"]:
    lo, ln = per_match_loss(p_old[m], y), per_match_loss(p_new[m], y)
    d = ln - lo
    se = d.std(ddof=1) / np.sqrt(len(d))
    bd = np.array([d[blk == b].mean() for b in range(1, N_BLOCKS + 1)])
    print(f"\n=== {m} ===")
    print(f"{'block':<7}{'old':>8}{'new':>8}{'new-old':>9}")
    for b in range(1, N_BLOCKS + 1):
        print(f"{b:<7}{lo[blk == b].mean():>8.3f}{ln[blk == b].mean():>8.3f}{bd[b - 1]:>+9.3f}")
    print(f"POOLED old {lo.mean():.3f} | new {ln.mean():.3f} | new-old {d.mean():+.4f} ± {2 * se:.4f} "
          f"({verdict(d.mean(), se)})")
    print(f"accuracy old {(p_old[m].argmax(1) == y).mean():.3f} | new {(p_new[m].argmax(1) == y).mean():.3f}")
    print(f"avg draw prob old {p_old[m][:, 1].mean():.3f} | new {p_new[m][:, 1].mean():.3f}")
    if m == "blend":
        c1 = d.mean() < -2 * se
        c2 = int((bd < 0).sum()) >= 5
        c3 = bool((bd <= 0.01).all())
        print("\nPRE-REGISTERED RULE (blend only):")
        print(f"  better than -2 SE:            {c1}")
        print(f"  better in at least 5 of 6:    {c2}  ({int((bd < 0).sum())} of 6)")
        print(f"  no block worse by more than 0.01: {c3}")
        print("  DECISION:", "ADOPT (eligible, production still waits)" if (c1 and c2 and c3)
              else "NOT ADOPTED")