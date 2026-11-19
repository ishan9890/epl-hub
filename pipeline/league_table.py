import json
from datetime import datetime, timezone

import pandas as pd

from .config import CURRENT, RAW, WEB_DATA


def build_table():
    df = pd.read_csv(RAW / f"E0_{CURRENT}.csv").dropna(subset=["FTR"])

    home = pd.DataFrame({"team": df["HomeTeam"], "gf": df["FTHG"], "ga": df["FTAG"]})
    away = pd.DataFrame({"team": df["AwayTeam"], "gf": df["FTAG"], "ga": df["FTHG"]})
    games = pd.concat([home, away], ignore_index=True)

    games["won"] = (games["gf"] > games["ga"]).astype(int)
    games["drawn"] = (games["gf"] == games["ga"]).astype(int)
    games["lost"] = (games["gf"] < games["ga"]).astype(int)

    table = games.groupby("team").agg(
        played=("gf", "size"),
        won=("won", "sum"),
        drawn=("drawn", "sum"),
        lost=("lost", "sum"),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
    ).reset_index()

    table["gd"] = table["gf"] - table["ga"]
    table["points"] = table["won"] * 3 + table["drawn"]
    table = table.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True)
    table.insert(0, "pos", table.index + 1)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": json.loads(table.to_json(orient="records")),
    }
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    (WEB_DATA / "league_table.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote league table with {len(table)} teams")


if __name__ == "__main__":
    build_table()