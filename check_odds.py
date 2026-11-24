import sys

import numpy as np
import pandas as pd

from pipeline.config import RAW, SEASONS
from pipeline.download import read_football_csv
from pipeline.evaluate import (FITTERS, MIN_TRAIN, N_BLOCKS, load_labeled,
                               per_match_loss, verdict)

# Pre-registered: the PRIMARY source is Bet365 closing odds. The rest are secondary.
SOURCES = {
    "B365 closing (PRIMARY)": ("B365CH", "B365CD", "B365CA"),
    "Market avg closing": ("AvgCH", "AvgCD", "AvgCA"),
    "B365 opening": ("B365H", "B365D", "B365A"),
}
KEYS = ["Date", "HomeTeam", "AwayTeam"]

# 1. Our matches (same ones evaluate.py uses)
labeled, features = load_labeled()
missing = [k for k in KEYS if k not in labeled.columns]
if missing:
    print("load_labeled() has no column(s):", missing)
    print("Columns available:", list(labeled.columns))
    sys.exit(1)
labeled["Date"] = pd.to_datetime(labeled["Date"]).dt.normalize()

# 2. Odds, loaded separately from the raw CSVs (features untouched)
odds_cols = [c for cols in SOURCES.values() for c in cols]
parts = []
for season in SEASONS:
    df = read_football_csv(RAW / f"E0_{season}.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.normalize()
    parts.append(df[KEYS + odds_cols])
odds = pd.concat(parts, ignore_index=True)
print("duplicate match keys in odds:", int(odds.duplicated(KEYS).sum()))

# 3. Merge. 'left' keeps our row order; validate stops silent duplication.
merged = labeled[KEYS + ["target"]].merge(odds, on=KEYS, how="left",
                                          validate="many_to_one")
assert len(merged) == len(labeled), "merge changed the row count"
test = merged.iloc[MIN_TRAIN:].reset_index(drop=True)
y = test["target"].to_numpy().astype(int)
print("test matches:", len(test))
print("test rows with NO odds row at all:", int(test[odds_cols].isna().all(axis=1).sum()))

# 4. Regenerate the blend's out-of-sample probabilities (same blocks as evaluate.py)
X = labeled[features].to_numpy()
yl = labeled["target"].to_numpy()
n = len(labeled)
blocks = np.array_split(np.arange(MIN_TRAIN, n), N_BLOCKS)
block_id = np.zeros(n - MIN_TRAIN, dtype=int)
blend = np.zeros((n - MIN_TRAIN, 3))
for i, idx in enumerate(blocks, start=1):
    start, stop = int(idx[0]), int(idx[-1]) + 1
    p_logit = FITTERS["logit"](X[:start], yl[:start], X[start:stop])
    p_xgb = FITTERS["xgb_small"](X[:start], yl[:start], X[start:stop])
    blend[start - MIN_TRAIN:stop - MIN_TRAIN] = (p_logit + p_xgb) / 2
    block_id[start - MIN_TRAIN:stop - MIN_TRAIN] = i
    print(f"  block {i}/{N_BLOCKS} done")

blend_loss = per_match_loss(blend, y)
print(f"\nSanity: blend pooled log loss {blend_loss.mean():.3f} on {len(y)} matches "
      f"(expect 0.994 on 1329)")

# 5. Score each odds source
for label, (h, d, a) in SOURCES.items():
    raw = test[[h, d, a]].to_numpy(dtype=float)
    ok = np.isfinite(raw).all(axis=1) & (raw > 1.0).all(axis=1)
    print(f"\n=== {label} ===  usable rows: {int(ok.sum())} of {len(ok)}  "
          f"(missing/invalid: {int((~ok).sum())})")
    inv = 1.0 / raw[ok]
    overround = inv.sum(axis=1)
    P = inv / overround[:, None]          # normalise the margin away
    print(f"average overround: {overround.mean():.3f}")

    lo = per_match_loss(P, y[ok])
    lb = blend_loss[ok]
    print(f"{'block':<7}{'n':>5}{'odds':>9}{'blend':>9}")
    for i in range(1, N_BLOCKS + 1):
        m = block_id[ok] == i
        print(f"{i:<7}{int(m.sum()):>5}{lo[m].mean():>9.3f}{lb[m].mean():>9.3f}")
    diff = lo - lb                         # negative = odds better than blend
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"POOLED odds {lo.mean():.3f} | blend {lb.mean():.3f} | "
          f"odds minus blend {diff.mean():+.3f} ± {2 * se:.3f}  "
          f"({verdict(diff.mean(), se)} = odds vs blend)")
    print(f"accuracy odds {(P.argmax(axis=1) == y[ok]).mean():.3f} | "
          f"blend {(blend[ok].argmax(axis=1) == y[ok]).mean():.3f}")
    print(f"avg draw prob: odds {P[:, 1].mean():.3f} | blend {blend[ok][:, 1].mean():.3f} "
          f"| actual {(y[ok] == 1).mean():.3f}")