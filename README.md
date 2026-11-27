# EPL Hub ⚽

A small end-to-end machine learning project that predicts the **next Premier League matchweek** (home win / draw / away win probabilities) and shows them on a static website, along with a league table and a results tracker that scores every prediction against what actually happened. Built as a learning project: Python for data and ML, plain JavaScript for the site.

**Live site:** https://epl-hub-m1bk.onrender.com

> **Not betting advice.** Football is noisy. These are probabilities from a simple model, not guarantees. This project is not affiliated with the Premier League.

![EPL Hub screenshot](docs/screenshot.png)

## How it works

```
PYTHON PIPELINE (runs weekly, on your computer)           STATIC WEBSITE (in the browser)
  download results + fixtures                               reads web/data/*.json
  -> Elo ratings + rolling-form features (no leakage)       -> Predictions, Table and
  -> train a blend of two models                               Results tabs
  -> write predictions to JSON, archive them  ------------->
  -> score archived predictions against real results
```

The JSON files in `web/data/` are the only link between the two halves. New projects plug in by adding a pipeline script, a JavaScript module that exports `render(container, data)`, and an entry in `web/data/projects.json`.

## The model

**Features (23)**, all computed using only information available before kickoff:

- Rolling averages over each team's previous 5 games: goals scored and conceded, shots, shots on target, shots conceded, and points
- Rest days since the last match
- An **Elo rating** for each team, calculated from the match results already downloaded (no extra data source). Every team enters at 1450 the first time it appears in the data, with K = 20 and a home advantage of 65 rating points. The rating used for a match is the one **before** that match is played. There is no margin-of-victory term and no regression to the mean between seasons, and a team that leaves the data keeps its last rating. These settings were fixed in advance, not tuned.
- Home-minus-away differences of the above

**Model:** an equal-weight blend of logistic regression and a cautious XGBoost (shallow trees, few of them, with a minimum amount of data per leaf). In the calibration tables below, the blend is less overconfident than either logistic regression or the original XGBoost settings.

> **Status:** the Elo blend is live. It made the predictions for the 27-29 November round. The earlier 21-23 November round was predicted by the previous 20-feature blend, and its archived predictions are kept unchanged, so the tracker says which model made each round.

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
- All four models with Elo beat the baseline clearly. The head-to-head tests below show which of their differences are detectable and which are not.
- Adding Elo improved the blend in all six blocks. On the earlier 1,319-match set, the blend went from 0.508 accuracy / 1.019 log loss without Elo to 0.532 / 0.993 with it, and a paired test gave -0.0264 ± 0.0123.
- Head-to-head tests on the same 1,329 matches (paired, with a stricter 2.39-standard-error bar because three comparisons were made): the blend beats the cautious XGBoost alone (-0.0065 log loss, better in all six blocks) but cannot be distinguished from logistic regression alone (-0.0012 ± 0.0051, better in only 3 of 6 blocks). The two members pick the same winner in 93% of matches. The blend stays in production because it was never significantly worse and because swapping models before live evidence exists would break the tracker's continuity.
- Temperature scaling (one-parameter softening of the blend's probabilities, fitted walk-forward) improved log loss by 0.0012 ± 0.0028 over 1,107 matches. That is not distinguishable from zero, so no calibration step was added.
- Adding five earlier seasons (2015/16 to 2019/20) to the training data improved the blend's log loss by 0.0048 ± 0.0070, which cannot be distinguished from zero (only 3 of 6 blocks improved), so it was not adopted. The test matches were fixed in advance, and both setups trained only on earlier matches.
- A diagnostic (`python -m pipeline.promoted_check`) compared expected and actual points for newly arrived teams over 387 out-of-sample team-matches. They earned 0.10 fewer points per match than predicted (±0.11), which cannot be distinguished from zero, so no promoted-team fix was made.
- Benchmark against bookmakers (measurement only, never used as a feature): Bet365 closing odds, with the margin normalised away, scored 0.964 log loss on the same 1,329 matches against the blend's 0.994 (difference -0.029 ± 0.011, better in all six blocks). The blend covers roughly 70% of the distance from the baseline (1.072) to the closing odds, but not all of it; the market also sees late news such as line-ups.

**Honest caveats**

- The blend's edge over the baseline has shrunk across the six blocks (0.128 in the first, 0.034 in the last). It is still positive, but six blocks cannot say whether that is noise, drift or stale features.
- The blend underpredicts draws (about 22% predicted against 25% actual). Temperature scaling cannot fix that, so it is being watched on the live tracker.
- One calibration bucket is somewhat overconfident: predictions of 50-60% happened 48.5% of the time (334 matches).
- Newly promoted teams start at an Elo of 1450 and have little history, so their predictions are less informed.
- The 1,329 test matches have already shaped several decisions (which features and models to keep). Only the live tracker is a fully clean test.
- Ten matches per round is far too few to judge a model. The live tracker needs dozens of rounds.

## Results tracker

Each round's predictions are saved to `data/history/` **before kickoff** and frozen once the round's first match date arrives (the freeze is by date, not by clock time). After the matches, the **Results** tab scores them (correct or not, and log loss) against a baseline. Earlier rounds from before tracking began show real scores only, labelled "No predictions made", because predicting matches the model was trained on would be meaningless.

The first tracked round (21-23 November, made by the earlier 20-feature blend) went 4 of 10 correct, with a log loss of 1.070 against a baseline of 1.048. With ten matches the standard error is about ±0.155 in log loss, so this says nothing about the model either way.

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
│   ├── promoted_check.py  # experiment: promoted-team diagnostic
│   ├── calibrate_check.py # experiment: temperature scaling
│   ├── head_to_head.py    # experiment: paired tests between blend members
│   ├── league_table.py    # build the league table JSON
│   └── run_weekly.py      # one command that runs everything
├── web/                   # JavaScript: the website
│   ├── index.html
│   ├── css/style.css
│   ├── js/                # main.js, api.js, teams.js (club codes and kit colours), projects/ (one module per tab)
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
git clone https://github.com/ishan9890/epl-hub.git
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

1. Wait until the previous round's results are in `data/raw/E0_<season>.csv` (usually a day or two after the last game). Check the played-match count for the season: it should have grown by 10 per round.
2. `python -m pipeline.download`, then `python -m pipeline.tracker` to preview the scored round.
3. `python -m pipeline.run_weekly --skip-download`, before the next round's first kickoff.
4. Check the printed fixtures (about 10 matches, no unknown-team warnings). Confirm that the new archive in `data/history/` matches `web/data/match_predictor.json` (`python compare_snapshot.py`) and that older archive files are not modified (`git status`).
5. Commit and push the updated `web/data/*.json` and `data/history/` files. The static site redeploys automatically.

Useful flag: `--skip-download` reuses the CSVs already in `data/raw/`. To re-check the model after changing features or settings, run `python -m pipeline.evaluate`.

## Data sources

- Match results and statistics: [football-data.co.uk](https://www.football-data.co.uk)
- Bookmaker odds, from the same football-data.co.uk files: used only for the benchmark above, never as a model input
- Upcoming fixtures: the Fantasy Premier League API (unofficial and undocumented, so it may change without notice)
- Elo ratings: calculated in this project from the match results
- Typeface: Archivo, loaded from Google Fonts

Raw data is not stored in this repository; the pipeline downloads it.

## Roadmap

- [x] Results tracker with archived pre-match predictions
- [x] Expanding-window evaluation with calibration tables
- [x] Blend of logistic regression and cautious XGBoost
- [x] Elo ratings built from the results
- [x] Bookmaker odds as a benchmark
- [x] Site redesign with team colours
- [x] Tested and not adopted (no detectable gain): temperature scaling, five more seasons of history, a promoted-team fix
- [ ] League table form guide and a redesigned Results tab
- [ ] Poisson / Dixon-Coles goal model as an extra blend member
- [ ] xG data
- [ ] Revisit the draw underprediction only if it persists over months of live rounds
- [ ] Elo charts

## License

MIT. See [LICENSE](LICENSE).