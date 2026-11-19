import pandas as pd
from .config import RAW, SEASONS, WINDOW

STATS = ["gf", "ga", "shots", "sot", "shots_against", "points"]
COLS = ["Date", "season", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST"]


def parse_dates(series):
    return pd.to_datetime(series, dayfirst=True, format="mixed")


def load_played():
    """All matches that have already finished, oldest first."""
    frames = []
    for s in SEASONS:
        path = RAW / f"E0_{s}.csv"
        if path.exists():
            d = pd.read_csv(path)
            d["season"] = s
            frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = parse_dates(df["Date"])
    df = df[COLS].dropna(subset=["FTR"])
    return df.sort_values("Date").reset_index(drop=True)


def load_upcoming(days_window=4):
    """Next matchweek. Priority: manual file > FPL file > football-data fixtures."""
    manual = RAW / "fixtures_manual.csv"
    fpl = RAW / "fixtures_fpl.csv"

    if manual.exists():
        fx = pd.read_csv(manual, encoding="utf-8-sig")
        use_window = True
        print("fixtures source: fixtures_manual.csv")
    elif fpl.exists():
        fx = pd.read_csv(fpl)
        use_window = False          # FPL already gives exactly one gameweek
        print("fixtures source: FPL")
    else:
        fx = pd.read_csv(RAW / "fixtures.csv")
        fx = fx[fx["Div"] == "E0"].copy()
        use_window = True
        print("fixtures source: football-data")

    if fx.empty:
        print("No upcoming Premier League fixtures found (international break?).")
        return None

    fx["Date"] = parse_dates(fx["Date"])
    fx = fx[fx["Date"] >= pd.Timestamp.today().normalize()]
    if fx.empty:
        print("All listed fixtures are in the past. Re-download or update the fixtures file.")
        return None

    if use_window:
        first = fx["Date"].min()
        fx = fx[fx["Date"] <= first + pd.Timedelta(days=days_window)]
    return fx[["Date", "HomeTeam", "AwayTeam"]].reset_index(drop=True)


def check_team_names(upcoming, played):
    """Warn if a fixture uses a team name the results data has never seen."""
    known = set(played["HomeTeam"]) | set(played["AwayTeam"])
    names = set(upcoming["HomeTeam"]) | set(upcoming["AwayTeam"])
    unknown = names - known
    if unknown:
        print("WARNING: unknown team names:", sorted(unknown))
        print("Known names:", sorted(known))

def build_features(played, upcoming=None):
    df = played.copy()

    # THE TRICK: append upcoming fixtures as rows with no result yet
    if upcoming is not None:
        up = upcoming.copy()
        up["season"] = df["season"].iloc[-1]
        df = pd.concat([df, up], ignore_index=True)

    df = df.sort_values("Date", kind="stable").reset_index(drop=True)

    # one row per team per match
    home = pd.DataFrame({
        "Date": df["Date"], "match_id": df.index, "team": df["HomeTeam"], "is_home": 1,
        "gf": df["FTHG"], "ga": df["FTAG"],
        "shots": df["HS"], "sot": df["HST"], "shots_against": df["AS"],
    })
    away = pd.DataFrame({
        "Date": df["Date"], "match_id": df.index, "team": df["AwayTeam"], "is_home": 0,
        "gf": df["FTAG"], "ga": df["FTHG"],
        "shots": df["AS"], "sot": df["AST"], "shots_against": df["HS"],
    })
    tm = pd.concat([home, away]).sort_values(["team", "Date"]).reset_index(drop=True)
    tm["points"] = (tm["gf"] > tm["ga"]) * 3 + (tm["gf"] == tm["ga"]) * 1

    # rolling averages of each team's PREVIOUS matches
    for s in STATS:
        tm[f"{s}_avg"] = (
            tm.groupby("team")[s]
              .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=3).mean())
        )
    tm["rest_days"] = tm.groupby("team")["Date"].diff().dt.days.clip(upper=14)

    # back to one row per match
    feat_cols = [f"{s}_avg" for s in STATS] + ["rest_days"]
    h = tm[tm["is_home"] == 1].set_index("match_id")[feat_cols].add_prefix("home_")
    a = tm[tm["is_home"] == 0].set_index("match_id")[feat_cols].add_prefix("away_")
    data = df.join(h).join(a)

    # difference features: how much better is the home side?
    for s in STATS:
        data[f"diff_{s}"] = data[f"home_{s}_avg"] - data[f"away_{s}_avg"]

    data["target"] = data["FTR"].map({"H": 0, "D": 1, "A": 2})
    features = [c for c in data.columns if c.startswith(("home_", "away_", "diff_"))]
    return data, features        