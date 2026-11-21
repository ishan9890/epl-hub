import pandas as pd

NEW_TEAM = 1450
K = 20
HOME_ADV = 65


def expected_home(home_rating, away_rating):
    diff = home_rating + HOME_ADV - away_rating
    return 1 / (1 + 10 ** (-diff / 400))


def compute_elo(matches):
    matches = matches.copy()
    matches["Date"] = pd.to_datetime(matches["Date"])
    matches = matches.sort_values("Date", kind="stable")

    ratings = {}
    out = []

    for row in matches.itertuples():
        home = ratings.get(row.HomeTeam, NEW_TEAM)
        away = ratings.get(row.AwayTeam, NEW_TEAM)

        out.append((row.Date, row.HomeTeam, row.AwayTeam, home, away))

        if pd.isna(row.FTHG) or pd.isna(row.FTAG):
            continue

        if row.FTHG > row.FTAG:
            score = 1.0
        elif row.FTHG == row.FTAG:
            score = 0.5
        else:
            score = 0.0

        change = K * (score - expected_home(home, away))
        ratings[row.HomeTeam] = home + change
        ratings[row.AwayTeam] = away - change

    return pd.DataFrame(
        out, columns=["Date", "HomeTeam", "AwayTeam", "elo_home", "elo_away"]
    )