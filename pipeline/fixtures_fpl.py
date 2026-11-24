import pandas as pd
import requests

from .config import RAW

BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (epl-hub learning project)"}
OUT = RAW / "fixtures_fpl.csv"

# FPL name -> football-data name (only where they differ)
NAME_MAP = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
}


def get_json(path):
    response = requests.get(f"{BASE}/{path}", headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def download_fpl_fixtures():
    """Save the NEXT gameweek's fixtures that have not kicked off yet to data/raw/fixtures_fpl.csv."""
    RAW.mkdir(parents=True, exist_ok=True)

    teams = get_json("bootstrap-static/")["teams"]
    names = {t["id"]: NAME_MAP.get(t["name"], t["name"]) for t in teams}

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for f in get_json("fixtures/"):
        if f["finished"] or f["event"] is None or f["kickoff_time"] is None:
            continue
        kickoff = pd.to_datetime(f["kickoff_time"], utc=True)
        if kickoff <= now:
            continue          # already kicked off, even if FPL has not marked it finished
        rows.append({
            "Date": kickoff.strftime("%d/%m/%Y"),
            "HomeTeam": names[f["team_h"]],
            "AwayTeam": names[f["team_a"]],
            "gameweek": f["event"],
        })

    if not rows:
        OUT.unlink(missing_ok=True)
        print("FPL: no upcoming fixtures found")
        return

    df = pd.DataFrame(rows)
    df = df[df["gameweek"] == df["gameweek"].min()]
    df.to_csv(OUT, index=False)
    print(f"saved FPL fixtures: gameweek {int(df['gameweek'].iloc[0])}, {len(df)} matches")


if __name__ == "__main__":
    download_fpl_fixtures()
    print(pd.read_csv(OUT) if OUT.exists() else "nothing saved")