import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from .config import HISTORY, MODELS, WEB_DATA
from .features import build_features, check_team_names, load_played, load_upcoming
from .models import blend_proba

LABELS = ["home", "draw", "away"]


def write_json(out):
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    path = WEB_DATA / "match_predictor.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def archive_snapshot(out):
    """Keep a pre-match copy of each round's predictions in data/history/."""
    if not out["matches"]:
        return

    first = min(m["date"] for m in out["matches"])
    path = HISTORY / f"pred_{first}.json"
    started = pd.Timestamp.today().normalize() >= pd.Timestamp(first)

    if path.exists() and started:
        print(f"archive: kept existing {path.name} (the round has already started)")
        return
    if started:
        print("WARNING: the round has already started, so this snapshot may be partial.")

    HISTORY.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"archived predictions to {path}")


def predict_next_matchweek():
    bundle = joblib.load(MODELS / "match_model.joblib")
    members, features = bundle["members"], bundle["features"]
    metrics = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))

    played = load_played()
    upcoming = load_upcoming()

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": metrics,
        "message": "",
        "matches": [],
    }

    if upcoming is None:
        out["message"] = ("No Premier League fixtures are available right now "
                          "(international break?). Check back after the next update.")
        path = write_json(out)
        print("no fixtures; wrote an empty page file:", path)
        return

    check_team_names(upcoming, played)
    print(upcoming)

    data, _ = build_features(played, upcoming)
    rows = data[data["FTR"].isna()]
    missing = int(rows[features].isna().any(axis=1).sum())
    if missing:
        print(f"WARNING: {missing} fixture(s) have missing features (too little history).")

    probs = blend_proba(members, rows[features])

    for (_, r), p in zip(rows.iterrows(), probs):
        out["matches"].append({
            "date": r["Date"].strftime("%Y-%m-%d"),
            "home": r["HomeTeam"],
            "away": r["AwayTeam"],
            "probs": {k: round(float(v), 3) for k, v in zip(LABELS, p)},
            "pick": LABELS[int(p.argmax())],
        })

    path = write_json(out)
    archive_snapshot(out)
    print(f"wrote {len(out['matches'])} predictions to {path}")


if __name__ == "__main__":
    predict_next_matchweek()