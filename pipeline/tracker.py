import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import HISTORY, WEB_DATA
from .features import load_played

LABELS = ["home", "draw", "away"]
RESULT_NAME = {"H": "home", "D": "draw", "A": "away"}
MIN_P = 1e-6   # stops log(0) from blowing up


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


def save(out, out_file):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")


def build_tracker(history_dir=HISTORY, out_file=None):
    out_file = out_file or (WEB_DATA / "tracker.json")
    Path(history_dir).mkdir(parents=True, exist_ok=True)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "message": "",
        "summary": None,
        "rounds": [],
    }

    preds = load_snapshots(history_dir)
    if preds.empty:
        out["message"] = "No archived predictions yet. Rounds appear here once they are saved."
        save(out, out_file)
        print("tracker: no snapshots yet; wrote an empty tracker file")
        return out

    played = load_played()
    preds = attach_results(preds, played)

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
            "label": f"{first:%d %b} to {last:%d %b %Y}",
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

    out["summary"] = summary
    out["rounds"] = list(reversed(rounds))          # newest first
    save(out, out_file)
    print("tracker:", json.dumps(summary))
    return out


if __name__ == "__main__":
    build_tracker()