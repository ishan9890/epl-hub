import argparse

from .download import download
from .predict import predict_next_matchweek
from .train import train
from .league_table import build_table
from .tracker import build_tracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly EPL prediction update")
    parser.add_argument("--all", action="store_true",
                        help="download every season (first run only)")
    parser.add_argument("--skip-download", action="store_true",
                        help="use the CSV files already in data/raw")
    args = parser.parse_args()

    if not args.skip_download:
            download(only_current=not args.all)
    train()
    predict_next_matchweek()
    build_table()
    build_tracker()
    print("\nweekly update finished.")
  
    