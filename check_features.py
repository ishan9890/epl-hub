from pipeline.features import load_played, build_features

played = load_played()
print("matches loaded:", len(played))
print(played.tail(3)[["Date", "HomeTeam", "AwayTeam", "FTR"]])

data, features = build_features(played)
print("number of features:", len(features))
print(features)
print(data[features].tail(3).round(2).T)

# --- verify by hand: Arsenal's goals scored, last match vs previous 5 ---
team = "Arsenal"
mine = played[(played.HomeTeam == team) | (played.AwayTeam == team)].tail(6)
last = mine.iloc[-1]
prev5 = mine.iloc[:-1]

goals = [r.FTHG if r.HomeTeam == team else r.FTAG for r in prev5.itertuples()]
print("\nmy manual average of the previous 5 games:", sum(goals) / 5)

row = data[(data.Date == last.Date) & ((data.HomeTeam == team) | (data.AwayTeam == team))].iloc[0]
col = "home_gf_avg" if row.HomeTeam == team else "away_gf_avg"
print("the value in the feature table:          ", round(row[col], 4))