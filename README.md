# EPL Hub ⚽

A small end-to-end machine learning project that predicts the **next Premier League matchweek** (home win / draw / away win probabilities) and shows them on a static website, along with a league table and a results tracker that scores every prediction against what actually happened. Built as a learning project: Python for data and ML, plain JavaScript for the site.

**Live site:** https://epl-hub-m1bk.onrender.com

> **Not betting advice.** Football is noisy. These are probabilities from a simple model, not guarantees. This project is not affiliated with the Premier League.

![EPL Hub screenshot](docs/screenshot.png)

## How it works

```
PYTHON PIPELINE (runs weekly, on your computer)           STATIC WEBSITE (in the browser)
  download results + fixtures                               reads web/data/*.json
  -> Elo ratings + rolling-form features (no leakage)       -> match cards, league table,
  -> train a blend of two models                               results tracker
  -> write predictions to JSON, archive them  ------------->
  -> score archived predictions against real results
```

The JSON files in `web/data/` are the only link between the two halves. New projects plug in by adding a pipeline script, a JavaScript module that exports `render(container, data)`, and an entry in `web/data/projects.json`.

## The model

**Features (23)**, all computed using only information available before kickoff:

- Rolling averages over each team's previous 5 games: goals scored and conceded, shots, shots on target, shots conceded, and points
- Rest days since the last match
- An **Elo rating** for each team, calculated from the match results already downloaded (no extra data source). Every team starts at 1450 in September 2020, with Every team starts at 1450 (a newly seen team also enters at 1450), with K = 20 and a home advantage of 65 rating points.. The rating used for a match is the one **before** that match is played
- Home-minus-away differences of the above

**Model:** an equal-weight blend of logistic regression and a cautious XGBoost (shallow trees, few of them, with a minimum amount of data per leaf). Averaging the two keeps the probabilities better calibrated than XGBoost alone.

> **Status:** the Elo blend was accepted from the evaluation below and goes live with the first weekly update after the 21-23 November round. Until then the site shows the earlier 20-feature blend, which is what the archived 21-23 November predictions use.

## Evaluation

All numbers come from `python -m pipeline.evaluate`: an expanding-window test over **1,329 matches** (April 2023 to 23 November 2026, 6 consecutive blocks). The model trains on everything before each block and predicts the block. Features only look backward, so there is no peeking.

| Model | Accuracy | Log loss (lower is better) |
|---|---|---|
| Baseline (historical home/draw/away rates) | 0.436 | 1.072 |
| Logistic regression, with Elo | 0.527 | 0.995 |
| XGBoost (original settings), with Elo | 0.515 | 1.026 |
| Cautious XGBoost, with Elo | 0.536 | 1.000 |
| **Blend, with Elo** | **0.529** | **0.994** |

- The blend beats the baseline in log loss by 0.078, with a 2-standard-error range of ±0.023.
- All four models with Elo beat the baseline clearly and are statistically close to each other. No head-to-head test has been run, so the choice among them is not a strong claim.
- The blend's edge over the baseline has shrunk in recent blocks (about 0.128 in the first block, 0.034 in the last). It is still positive, but the live tracker is the cleaner test.
- Adding Elo improved the blend in all six blocks. On the earlier 1,319-match set, the blend went from 0.508 accuracy / 1.019 log loss without Elo to 0.532 / 0.993 with it, and a paired test gave -0.0264 ± 0.0123.
- Calibration is imperfect at the extremes. For the blend, predictions of 50-60% happened 48.5% of the time (334 matches), and logistic regression and the original XGBoost are more overconfident than the blend.
- Temperature scaling (one-parameter softening of the blend's probabilities, fitted walk-forward) improved log loss by 0.0012 ± 0.0028 over 1,107 matches. That is not distinguishable from zero, so no calibration step was added. The blend underpredicts draws slightly (22% predicted against 25% actual); this is being watched on the live tracker.
- Benchmark against bookmakers (measurement only, never used as a feature): Bet365 closing odds, with the margin normalised away, scored 0.964 log loss on the same 1,329 matches against the blend's 0.994 (difference -0.029 ± 0.011, better in all six blocks). The blend therefore closes most of the gap between the baseline (1.072) and the market, but not all of it; the market also sees late news such as line-ups.

**Honest caveats**

- The blend's edge over the baseline has shrunk over the last six blocks (roughly 0.12 down to 0.04), and Elo did not change that trend. Six blocks cannot say whether that is noise or a real decline.
- One calibration bucket (predictions of 50-60%) is somewhat overconfident: 0.545 predicted, 0.485 observed.
- Newly promoted teams start at an Elo of 1450 and have little history, so their predictions are less informed.
- Ten matches per round is far too few to judge a model. The live tracker needs dozens of rounds.
- A diagnostic (`python -m pipeline.promoted_check`) compared expected and actual points for newly arrived teams over 387 out-of-sample team-matches. They earned 0.10 fewer points per match than predicted (±0.11), which can't be distinguished from zero, so no promoted-team fix was made. Revisit with more seasons of data.
- Adding five older seasons (2015-16 to 2019-20) to the training data was tested as a pre-registered experiment on the same 1,329 test matches. The blend's log loss changed by -0.0048 ± 0.0070, inside the noise, and only 3 of 6 blocks improved, so it was not adopted.

## Results tracker

Each round's predictions are saved to `data/history/` **before kickoff** and frozen once the round's first match date arrives. After the matches, the **Results** tab scores them (correct or not, and log loss) against a baseline. Earlier rounds from before tracking began show real scores only, labelled "No predictions made", because predicting matches the model was trained on would be meaningless.

## Project structure

```
epl-hub/
├── pipeline/              # Python: runs weekly
│   ├── config.py          # paths and settings
│   ├── download.py        # fetch results and fixtures
│   ├── fixtures_fpl.py    # next-gameweek fixtures from the FPL API
│   ├── elo.py             # Elo ratings built from the results
│   ├── features.py        # feature engineering
│   ├── models.py          # model definitions (one place for all of them)
│   ├── train.py           # train the blend, save it
│   ├── predict.py         # write next-matchweek predictions, archive them
│   ├── tracker.py         # score archived predictions against results
│   ├── evaluate.py        # expanding-window evaluation and calibration
│   ├── league_table.py    # build the league table JSON
│   └── run_weekly.py      # one command that runs everything
├── web/                   # JavaScript: the website
│   ├── index.html
│   ├── css/style.css
│   ├── js/                # main.js, api.js, projects/ (one module per tab)
│   └── data/              # JSON written by Python, read by the site
├── data/
│   ├── history/           # archived pre-match predictions (committed)
│   └── raw/               # downloaded CSVs (not committed)
├── models/                # trained models and evaluation output (not committed)
├── check_*.py, show_predictions.py, compare_snapshot.py   # small helper scripts
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

1. Wait until the previous round's results are in `data/raw/E0_<season>.csv` (usually a day or two after the last game).
2. `python -m pipeline.run_weekly` before the next round's first kickoff.
3. Check the printed fixtures and warnings (unknown team names, missing features).
4. Commit and push the updated `web/data/*.json` and `data/history/` files. The static site redeploys automatically.

Useful flag: `--skip-download` reuses the CSVs already in `data/raw/`. To re-check the model after changing features or settings, run `python -m pipeline.evaluate`.

## Data sources

- Match results and statistics: [football-data.co.uk](https://www.football-data.co.uk)
- Upcoming fixtures: the Fantasy Premier League API (unofficial and undocumented, so it may change without notice)
- Elo ratings: calculated in this project from the match results

Raw data is not stored in this repository; the pipeline downloads it.

## Roadmap

- [x] Results tracker with archived pre-match predictions
- [x] Expanding-window evaluation with calibration tables
- [x] Blend of logistic regression and cautious XGBoost
- [x] Elo ratings built from the results
- [ ] Better handling of newly promoted teams (e.g. Championship results)
- [ ] More seasons of history
- [ ] Probability calibration and a better blend
- [ ] Poisson / Dixon-Coles goal model
- [ ] Bookmaker odds as a benchmark, then xG data
- [ ] UI polish: team colours, form guide, Elo charts

## License

MIT. See [LICENSE](LICENSE).