import pandas as pd

from pipeline.download import download
from pipeline.features import load_upcoming

download(only_current=True)

fx = pd.read_csv("data/raw/fixtures.csv")
e0 = fx[fx["Div"] == "E0"]

print("\nleagues in the file:", fx["Div"].nunique())
print("Premier League rows:", len(e0))
if len(e0):
    print(e0[["Date", "HomeTeam", "AwayTeam"]].to_string(index=False))

print("\nwhat the predictor would use:")
print(load_upcoming())