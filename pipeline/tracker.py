import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import CURRENT, HISTORY, WEB_DATA
from .features import load_played

LABELS = ["home", "draw", "away"]
RESULT_NAME = {"H": "home", "D": "draw", "A": "away"}
MIN_P = 1e-6   # stops log(0) from blowing up


def date_label(first, last):
    if first.date() == last.date():
        return f"{first:%d %b %Y}"
    return f"{first:%d %b} to {last:%d %b %Y}"


def load_snapshots(history_dir):
    rows = []
    for path in sorted(Path(history_dir).glob("pred_*.json")):
        snap = json.loads(path.read_text(encoding="utf-8"))
        for m in snap.get("matches", []):
            rows.append({
                "snapshot": path.stem,
                "date": pd.to_datetime(m["date"]),
                "home": m["home"],
                "away": m["away"],
                "p_home": float(m["probs"]["home"]),
                "p_draw": float(m["probs"]["draw"]),
                "p_away": float(m["probs"]["away"]),
                "pick": m["pick"],
                "model_name": snap.get("model", {}).get("model_name", "XGBoost (original)"),
            })
    return pd.DataFrame(rows)


def attach_results(preds, played):
    """Find the real result for each prediction (same teams, within 3 days)."""
    preds = preds.reset_index(drop=True)
    preds["pid"] = preds.index

    res = played[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].rename(
        columns={"Date": "result_date", "HomeTeam": "home", "AwayTeam": "away"}
    )
    cand = preds[["pid", "home", "away", "date"]].merge(res, on=["home", "away"], how="inner")
    cand = cand[(cand["result_date"] - cand["date"]).abs().dt.days <= 3]
    cand = cand.drop_duplicates("pid")[["pid", "FTHG", "FTAG", "FTR"]]
    return preds.merge(cand, on="pid", how="left")


def score_rounds(preds, played):
    """Score every saved round. Returns (rounds, summary)."""
    tot = {"n": 0, "correct": 0, "ll": 0.0, "b_correct": 0, "b_ll": 0.0, "pending": 0}
    rounds = []

    for snap, grp in preds.groupby("snapshot", sort=True):
        first, last = grp["date"].min(), grp["date"].max()

        # baseline = historical rates from matches BEFORE this round
        before = played[played["Date"] < first]["FTR"].map(RESULT_NAME)
        rates = before.value_counts(normalize=True)
        base = {k: float(rates.get(k, 1 / 3)) for k in LABELS}
        base_pick = max(base, key=base.get)

        matches = []
        n = correct = b_correct = 0
        ll = b_ll = 0.0

        for r in grp.sort_values("date").itertuples():
            probs = {"home": r.p_home, "draw": r.p_draw, "away": r.p_away}
            m = {
                "date": r.date.strftime("%Y-%m-%d"),
                "home": r.home,
                "away": r.away,
                "probs": probs,
                "pick": r.pick,
                "result": None,
                "score": None,
                "correct": None,
                "log_loss": None,
            }
            if isinstance(r.FTR, str):
                actual = RESULT_NAME[r.FTR]
                loss = -math.log(max(probs[actual], MIN_P))
                m.update(
                    result=actual,
                    score=f"{int(r.FTHG)}-{int(r.FTAG)}",
                    correct=(r.pick == actual),
                    log_loss=round(loss, 3),
                )
                n += 1
                correct += int(r.pick == actual)
                ll += loss
                b_correct += int(base_pick == actual)
                b_ll += -math.log(max(base[actual], MIN_P))
            matches.append(m)

        pending = len(matches) - n
        rounds.append({
            "snapshot": snap,
            "has_predictions": True,
            "model_name": grp["model_name"].iloc[0],
            "first_date": first.strftime("%Y-%m-%d"),
            "label": date_label(first, last),
            "matches": matches,
            "played": n,
            "pending": pending,
            "correct": correct,
            "log_loss": round(ll / n, 3) if n else None,
            "baseline_log_loss": round(b_ll / n, 3) if n else None,
        })
        tot["n"] += n
        tot["correct"] += correct
        tot["ll"] += ll
        tot["b_correct"] += b_correct
        tot["b_ll"] += b_ll
        tot["pending"] += pending

    n = tot["n"]
    summary = {"rounds": len(rounds), "matches_played": n, "matches_pending": tot["pending"]}
    if n:
        acc = tot["correct"] / n
        summary.update(
            accuracy=round(acc, 3),
            accuracy_se=round(math.sqrt(acc * (1 - acc) / n), 3),
            log_loss=round(tot["ll"] / n, 3),
            baseline_accuracy=round(tot["b_correct"] / n, 3),
            baseline_log_loss=round(tot["b_ll"] / n, 3),
        )
    return rounds, summary


def split_into_rounds(df):
    """Group matches into rounds: a new round starts when a team would play twice."""
    rounds, current, seen = [], [], set()
    for row in df.sort_values("Date", kind="stable").itertuples():
        teams = {row.HomeTeam, row.AwayTeam}
        if seen & teams:
            rounds.append(current)
            current, seen = [], set()
        current.append(row)
        seen |= teams
    if current:
        rounds.append(current)
    return rounds


def results_only_rounds(played, covered_pairs):
    """This season's earlier rounds: real scores, no predictions."""
    season = played[played["season"] == CURRENT]
    out = []
    for rnd in split_into_rounds(season):
        pairs = {(r.HomeTeam, r.AwayTeam) for r in rnd}
        if pairs & covered_pairs:
            continue            # this round has saved predictions, shown separately
        dates = [r.Date for r in rnd]
        first, last = min(dates), max(dates)
        out.append({
            "snapshot": None,
            "has_predictions": False,
            "first_date": first.strftime("%Y-%m-%d"),
            "label": date_label(first, last),
            "matches": [
                {
                    "date": r.Date.strftime("%Y-%m-%d"),
                    "home": r.HomeTeam,
                    "away": r.AwayTeam,
                    "score": f"{int(r.FTHG)}-{int(r.FTAG)}",
                    "result": RESULT_NAME[r.FTR],
                }
                for r in sorted(rnd, key=lambda r: r.Date)
            ],
        })
    return out


def save(out, out_file):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")


def build_tracker(history_dir=HISTORY, out_file=None):
    out_file = out_file or (WEB_DATA / "tracker.json")
    Path(history_dir).mkdir(parents=True, exist_ok=True)
    played = load_played()

    predicted, covered = [], set()
    summary = {"rounds": 0, "matches_played": 0, "matches_pending": 0}

    preds = load_snapshots(history_dir)
    if not preds.empty:
        preds = attach_results(preds, played)
        covered = set(zip(preds["home"], preds["away"]))
        predicted, summary = score_rounds(preds, played)

    earlier = results_only_rounds(played, covered)
    rounds = sorted(predicted + earlier, key=lambda r: r["first_date"], reverse=True)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "message": "" if rounds else "No results yet.",
        "tracking_started": min((r["first_date"] for r in predicted), default=None),
        "summary": summary,
        "rounds": rounds,
    }
    save(out, out_file)
    print(f"tracker: {len(predicted)} predicted round(s), {len(earlier)} results-only round(s)")
    print("tracker summary:", json.dumps(summary))
    return out


if __name__ == "__main__":
    build_tracker()