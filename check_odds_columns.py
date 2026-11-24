from pipeline.config import RAW, SEASONS
from pipeline.download import read_football_csv

# Candidate names. We are CHECKING which exist, not assuming.
CANDIDATES = ["B365H", "B365D", "B365A",
              "B365CH", "B365CD", "B365CA",
              "AvgH", "AvgD", "AvgA",
              "AvgCH", "AvgCD", "AvgCA"]

first = True
for season in SEASONS:
    path = RAW / f"E0_{season}.csv"
    if not path.exists():
        print(season, "FILE MISSING")
        continue
    df = read_football_csv(path)
    if first:
        print("ALL COLUMNS in", path.name)
        print(list(df.columns))
        print()
        first = False
    present = {c: int(df[c].notna().sum()) for c in CANDIDATES if c in df.columns}
    print(season, "rows:", len(df), "| non-empty odds columns:", present)