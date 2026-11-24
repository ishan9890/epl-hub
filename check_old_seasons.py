import pandas as pd

from pipeline.config import ROOT, RAW, SEASONS
from pipeline.download import BASE, read_football_csv
from pipeline.features import COLS, parse_dates

OLD_SEASONS = ["1516", "1617", "1718", "1819", "1920"]   # fixed in advance
OUT = ROOT / "data" / "_test_raw"                         # gitignored by data/_test*
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [c for c in COLS if c != "season"]   # season is added later by us
old_teams = set()

for s in OLD_SEASONS:
    print(f"\n=== season {s} ===")
    try:
        table = read_football_csv(BASE.format(s))
    except Exception as err:
        print("COULD NOT download:", err)
        continue
    table.to_csv(OUT / f"E0_{s}.csv", index=False)

    missing = [c for c in REQUIRED if c not in table.columns]
    print("rows:", len(table), "| missing required columns:", missing)
    if missing:
        continue

    nulls = {c: int(table[c].isna().sum()) for c in REQUIRED if table[c].isna().any()}
    print("required columns with empty cells:", nulls or "none")

    dates = parse_dates(table["Date"])
    print("dates:", dates.min().date(), "to", dates.max().date())
    teams = set(table["HomeTeam"]) | set(table["AwayTeam"])
    print("teams:", len(teams))
    old_teams |= teams

# compare team spellings with the seasons already in production
new_teams = set()
for s in SEASONS:
    path = RAW / f"E0_{s}.csv"
    if path.exists():
        d = pd.read_csv(path)
        new_teams |= set(d["HomeTeam"]) | set(d["AwayTeam"])

print("\nIn OLD seasons only:", sorted(old_teams - new_teams))
print("In CURRENT seasons only:", sorted(new_teams - old_teams))