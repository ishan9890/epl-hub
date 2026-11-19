from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
WEB_DATA = ROOT / "web" / "data"

SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526", "2627"]
CURRENT = "2627"
WINDOW = 5          # rolling form window (matches)