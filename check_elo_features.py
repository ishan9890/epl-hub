from pipeline.features import load_played, load_upcoming, build_features

played = load_played()
upcoming = load_upcoming()
data, features = build_features(played, upcoming)

expected = len(played) + (0 if upcoming is None else len(upcoming))
print("rows:", len(data), "(expected", expected, ")")
print("n features:", len(features), "(expected 23)")
print("elo columns in features:", [f for f in features if "elo" in f])
print("missing Elo values:", int(data[["home_elo", "away_elo", "diff_elo"]].isna().sum().sum()))

if upcoming is not None:
    print()
    cols = ["Date", "HomeTeam", "AwayTeam", "home_elo", "away_elo", "diff_elo"]
    print(data.tail(len(upcoming))[cols].round(0).to_string())