# EPL Hub ⚽

A small end-to-end machine learning project that predicts the **next Premier League matchweek** (home win / draw / away win probabilities) and displays them on a static website. Built as a learning project: Python for data and ML, plain JavaScript for the site.

> **Not betting advice.** Football is noisy. These are probabilities from a simple model, not guarantees. This project is not affiliated with the Premier League.

![EPL Hub screenshot](docs/screenshot.png)

## How it works

```
PYTHON PIPELINE (runs weekly, on your computer)       STATIC WEBSITE (in the browser)
  download results + fixtures                            reads web/data/*.json
  -> rolling-form features (no data leakage)             -> draws match cards and a league table
  -> train and evaluate models
  -> write predictions to JSON  ----------------------->
```

The JSON files in `web/data/` are the only link between the two halves. New projects plug in by adding a pipeline script, a JavaScript module that exports `render(container, data)`, and an entry in `web/data/projects.json`.

## Model performance

Back-test on the most recent 380 matches (numbers change as data updates):

| Model | Accuracy | Log loss (lower is better) |
|---|---|---|
| Baseline (historical class rates) | 0.400 | 1.099 |
| Logistic regression | 0.439 | 1.073 |
| XGBoost | 0.432 | 1.114 |

The current XGBoost model does not beat the baseline on log loss, so its probabilities are overconfident. Improving this is the main goal of the roadmap below.

## Project structure

```
epl-hub/
├── pipeline/            # Python: runs weekly
│   ├── config.py        # paths and settings
│   ├── download.py      # fetch results and fixtures
│   ├── fixtures_fpl.py  # next-gameweek fixtures from the FPL API
│   ├── features.py      # feature engineering
│   ├── train.py         # train, evaluate, save the model
│   ├── predict.py       # write next-matchweek predictions as JSON
│   ├── league_table.py  # build the league table JSON
│   └── run_weekly.py    # one command that runs everything
├── web/                 # JavaScript: the website
│   ├── index.html
│   ├── css/style.css
│   ├── js/              # main.js, api.js, projects/
│   └── data/            # JSON written by Python, read by the site
├── data/raw/            # downloaded CSVs (not committed)
├── models/              # trained models (not committed)
└── requirements.txt
```

## Getting started

Requires Python 3.11 or newer.

```bash
git clone https://github.com/YOUR-USERNAME/epl-hub.git
cd epl-hub
python -m venv venv
venv\Scripts\activate          # Windows (macOS/Linux: source venv/bin/activate)
python -m pip install -r requirements.txt
```

First run (downloads every season, trains, predicts):

```bash
python -m pipeline.run_weekly --all
```

View the site locally:

```bash
cd web
python -m http.server 8000
```

Then open <http://localhost:8000>.

## Weekly routine

1. `python -m pipeline.run_weekly` after the previous round's results are posted and before the next kickoff.
2. Check the printed fixtures and warnings (unknown team names, missing features).
3. Commit and push the updated `web/data/*.json` files.

Useful flag: `--skip-download` reuses the CSVs already in `data/raw/`.

## Data sources

- Match results and statistics: [football-data.co.uk](https://www.football-data.co.uk)
- Upcoming fixtures: the Fantasy Premier League API (unofficial and undocumented, so it may change without notice)

Raw data is not stored in this repository; the pipeline downloads it.

## Roadmap

- [ ] Elo ratings and longer history
- [ ] Better handling of newly promoted teams
- [ ] Probability calibration and a blended model
- [ ] Poisson / Dixon-Coles goal model
- [ ] Results tracker: how did last week's predictions do?
- [ ] xG data (Understat) and bookmaker odds as a benchmark
- [ ] UI polish: team colours, form guide, charts

## License

MIT. See [LICENSE](LICENSE).