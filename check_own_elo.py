import numpy as np
from pipeline.features import load_played
from pipeline.elo import compute_elo

played = load_played()
print("columns:", list(played.columns))
print("matches:", len(played))

elo = compute_elo(played)

# 1) First match: both teams should be at the starting value
print("\nfirst rows:")
print(elo.head(3))

# 2) Current ratings, from each team's latest match
latest = {}
for r in elo.itertuples():
    latest[r.HomeTeam] = r.elo_home
    latest[r.AwayTeam] = r.elo_away
top = sorted(latest.items(), key=lambda x: -x[1])
print("\ntop 6:", [(t, round(v)) for t, v in top[:6]])
print("bottom 4:", [(t, round(v)) for t, v in top[-4:]])

# 3) Leakage test: blank the last match's result and recompute
test = played.copy()
test["Date"] = test["Date"]
last_index = test.sort_values("Date", kind="stable").index[-1]
test.loc[last_index, ["FTHG", "FTAG"]] = np.nan
elo2 = compute_elo(test)

a = elo.iloc[-1][["elo_home", "elo_away"]].astype(float).values
b = elo2.iloc[-1][["elo_home", "elo_away"]].astype(float).values
print("\nleakage test (should say True):", bool(np.allclose(a, b)))