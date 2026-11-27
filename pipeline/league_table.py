import json
from datetime import datetime, timezone

import pandas as pd

from .config import CURRENT, RAW, WEB_DATA

FORM_GAMES = 5   # how many recent results to show in the form guide


def build_table():
    df = pd.read_csv(RAW / f"E0_{CURRENT}.csv").dropna(subset=["FTR"])
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True)

    home = pd.DataFrame({"team": df["HomeTeam"], "gf": df["FTHG"], "ga": df["FTAG"], "date": df["date"]})
    away = pd.DataFrame({"team": df["AwayTeam"], "gf": df["FTAG"], "ga": df["FTHG"], "date": df["date"]})
    games = pd.concat([home, away], ignore_index=True)

    games["won"] = (games["gf"] > games["ga"]).astype(int)
    games["drawn"] = (games["gf"] == games["ga"]).astype(int)
    games["lost"] = (games["gf"] < games["ga"]).astype(int)

    # one letter per game, then each team's last FORM_GAMES in date order (oldest first)
    games["result"] = "D"
    games.loc[games["won"] == 1, "result"] = "W"
    games.loc[games["lost"] == 1, "result"] = "L"
    ordered = games.sort_values("date", kind="stable")
    form = {team: list(g["result"].tail(FORM_GAMES)) for team, g in ordered.groupby("team")}

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

    rows = json.loads(table.to_json(orient="records"))
    for row in rows:
        row["form"] = form.get(row["team"], [])

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    (WEB_DATA / "league_table.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote league table with {len(table)} teams")


if __name__ == "__main__":
    build_table()