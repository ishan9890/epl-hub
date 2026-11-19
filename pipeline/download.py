import pandas as pd
from .config import RAW, SEASONS, CURRENT
from .fixtures_fpl import download_fpl_fixtures

BASE = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
FIXTURES = "https://www.football-data.co.uk/fixtures.csv"


def read_football_csv(url):
    """Download a CSV and clean up encoding problems."""
    try:
        table = pd.read_csv(url, encoding="utf-8-sig")   # utf-8-sig removes the BOM
    except UnicodeDecodeError:
        table = pd.read_csv(url, encoding="latin-1")
    # safety net: strip any leftover junk or spaces from column names
    table.columns = (
        table.columns.str.replace("ï»¿", "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return table


def download(only_current=True):
    RAW.mkdir(parents=True, exist_ok=True)
    seasons = [CURRENT] if only_current else SEASONS

    for s in seasons:
        try:
            table = read_football_csv(BASE.format(s))
            table.to_csv(RAW / f"E0_{s}.csv", index=False)
            print(f"saved season {s}: {len(table)} matches")
        except Exception as err:
            print(f"COULD NOT download season {s}: {err}")

        try:
            download_fpl_fixtures()
        except Exception as err:
            print(f"COULD NOT download FPL fixtures: {err}")


if __name__ == "__main__":
    download(only_current=False)