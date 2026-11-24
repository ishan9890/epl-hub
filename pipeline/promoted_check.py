import numpy as np
import pandas as pd

from .config import SEASONS
from .evaluate import FITTERS, MIN_TRAIN, N_BLOCKS, load_labeled
from .features import load_played


def out_of_sample_blend():
    """Same expanding-window blocks as evaluate.py. One row per test match."""
    labeled, features = load_labeled()
    X = labeled[features].to_numpy()
    y = labeled["target"].to_numpy()

    parts = []
    for idx in np.array_split(np.arange(MIN_TRAIN, len(labeled)), N_BLOCKS):
        start, stop = int(idx[0]), int(idx[-1]) + 1
        X_tr, y_tr, X_te = X[:start], y[:start], X[start:stop]
        p = (FITTERS["logit"](X_tr, y_tr, X_te) + FITTERS["xgb_small"](X_tr, y_tr, X_te)) / 2

        part = labeled.iloc[start:stop][["Date", "season", "HomeTeam", "AwayTeam", "target"]].copy()
        part["p_home"], part["p_draw"], part["p_away"] = p[:, 0], p[:, 1], p[:, 2]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def newly_arrived(played):
    """(season, team) pairs for teams that were NOT in the league the season before."""
    teams = {s: set(g["HomeTeam"]) | set(g["AwayTeam"]) for s, g in played.groupby("season")}
    flagged = set()
    for i in range(1, len(SEASONS)):
        now, before = SEASONS[i], SEASONS[i - 1]
        if now in teams and before in teams:
            for team in teams[now] - teams[before]:
                flagged.add((now, team))
    return flagged


def season_match_numbers(played):
    """(date, team) -> which match of the season this was for that team."""
    rows = pd.concat([
        played[["Date", "season", "HomeTeam"]].rename(columns={"HomeTeam": "team"}),
        played[["Date", "season", "AwayTeam"]].rename(columns={"AwayTeam": "team"}),
    ]).sort_values("Date", kind="stable")
    rows["n"] = rows.groupby(["season", "team"]).cumcount() + 1
    return {(r.Date, r.team): r.n for r in rows.itertuples()}


def team_view(preds, played):
    """Turn each match into two rows: one from each team's point of view."""
    flagged = newly_arrived(played)
    number = season_match_numbers(played)

    records = []
    for r in preds.itertuples():
        for side, team in (("home", r.HomeTeam), ("away", r.AwayTeam)):
            if side == "home":
                p_win, won = r.p_home, r.target == 0
            else:
                p_win, won = r.p_away, r.target == 2
            drew = r.target == 1

            actual = 3 * int(won) + int(drew)
            expected = 3 * p_win + r.p_draw

            if (r.season, team) in flagged:
                early = number[(r.Date, team)] <= 10
                group = "newly arrived: first 10 matches" if early else "newly arrived: later matches"
            else:
                group = "established teams"
            records.append({"group": group, "expected": expected, "actual": actual})
    return pd.DataFrame(records)


def row(name, g):
    gap = g["actual"] - g["expected"]
    two_se = 2 * gap.std(ddof=1) / np.sqrt(len(g))
    print(f"{name:<34}{len(g):>8}{g['expected'].mean():>10.2f}{g['actual'].mean():>8.2f}"
          f"{gap.mean():>+8.2f}{two_se:>8.2f}")
    return gap.mean(), two_se


def main():
    played = load_played()
    preds = out_of_sample_blend()
    print(f"{len(preds)} out-of-sample matches "
          f"({preds['Date'].min():%d %b %Y} to {preds['Date'].max():%d %b %Y})")

    print("\nTeams flagged as newly arrived, by season:")
    flagged = newly_arrived(played)
    for s in SEASONS:
        names = sorted(t for (season, t) in flagged if season == s)
        if names:
            print(f"  {s}: {', '.join(names)}")

    view = team_view(preds, played)
    print(f"\n{'group':<34}{'matches':>8}{'expected':>10}{'actual':>8}{'gap':>8}{'±2 SE':>8}")
    for name in ["newly arrived: first 10 matches", "newly arrived: later matches", "established teams"]:
        g = view[view["group"] == name]
        if len(g) > 1:
            row(name, g)

    new = view[view["group"].str.startswith("newly arrived")]
    gap, two_se = row("newly arrived: ALL", new)

    print("\nDecision:", end=" ")
    if gap < -two_se:
        print("the model OVERRATES newly arrived teams. A fix is worth trying.")
    elif gap > two_se:
        print("the model UNDERRATES newly arrived teams. Tell me before we fix anything.")
    else:
        print("no clear bias. Skip the fix and record this as a finding.")


if __name__ == "__main__":
    main()