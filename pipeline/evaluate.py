import json

import numpy as np
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from .config import MODELS
from .features import build_features, load_played
from .train import make_logit, make_xgb

N_BLOCKS = 6        # how many test blocks
MIN_TRAIN = 1000    # the first test block starts after this many matches
CLASSES = [0, 1, 2]
NAMES = ["baseline", "logit", "xgb", "xgb_small", "blend"]
EDGES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0001]


def make_xgb_small():
    return XGBClassifier(
        objective="multi:softprob",
        n_estimators=150,
        max_depth=2,
        min_child_weight=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def predict_baseline(X_tr, y_tr, X_te):
    rates = np.bincount(y_tr, minlength=3) / len(y_tr)
    return np.tile(rates, (len(X_te), 1))


def predict_logit(X_tr, y_tr, X_te):
    return make_logit().fit(X_tr, y_tr).predict_proba(X_te)


def predict_xgb(X_tr, y_tr, X_te):
    return make_xgb().fit(X_tr, y_tr).predict_proba(X_te)


def predict_xgb_small(X_tr, y_tr, X_te):
    return make_xgb_small().fit(X_tr, y_tr).predict_proba(X_te)


FITTERS = {
    "baseline": predict_baseline,
    "logit": predict_logit,
    "xgb": predict_xgb,
    "xgb_small": predict_xgb_small,
}


def load_labeled():
    played = load_played()
    data, features = build_features(played)
    labeled = data.dropna(subset=["target"] + features).copy()
    labeled["target"] = labeled["target"].astype(int)
    labeled = labeled.sort_values("Date").reset_index(drop=True)
    return labeled, features


def per_match_loss(P, y):
    """Log loss for each match: -ln(probability given to what actually happened)."""
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1.0))


def verdict(diff, se):
    if diff < -2 * se:
        return "better"
    if diff > 2 * se:
        return "worse"
    return "unclear"


def calibration(P, y):
    """Predicted probability vs how often it really happened, pooled over all 3 outcomes."""
    p = P.ravel()
    outcome = np.eye(3)[y].ravel()
    bucket = np.digitize(p, EDGES) - 1
    rows = []
    for b in range(len(EDGES) - 1):
        mask = bucket == b
        if mask.sum() < 20:
            continue
        rows.append({
            "range": f"{EDGES[b]:.1f}-{min(EDGES[b + 1], 1.0):.1f}",
            "count": int(mask.sum()),
            "predicted": round(float(p[mask].mean()), 3),
            "observed": round(float(outcome[mask].mean()), 3),
        })
    return rows


def evaluate():
    labeled, features = load_labeled()
    X = labeled[features].to_numpy()
    y = labeled["target"].to_numpy()
    n = len(labeled)
    blocks = np.array_split(np.arange(MIN_TRAIN, n), N_BLOCKS)
    print(f"{n} labeled matches; {N_BLOCKS} test blocks starting after the first {MIN_TRAIN}")

    pooled = {name: [] for name in NAMES}
    y_pooled, fold_rows = [], []

    for i, idx in enumerate(blocks, start=1):
        start, stop = int(idx[0]), int(idx[-1]) + 1
        X_tr, y_tr = X[:start], y[:start]
        X_te, y_te = X[start:stop], y[start:stop]

        probs = {name: fit(X_tr, y_tr, X_te) for name, fit in FITTERS.items()}
        probs["blend"] = (probs["logit"] + probs["xgb_small"]) / 2

        row = {
            "block": i,
            "train": int(len(y_tr)),
            "test": int(len(y_te)),
            "from": labeled["Date"].iloc[start].strftime("%Y-%m-%d"),
            "to": labeled["Date"].iloc[stop - 1].strftime("%Y-%m-%d"),
        }
        for name in NAMES:
            pooled[name].append(probs[name])
            row[name] = round(float(log_loss(y_te, probs[name], labels=CLASSES)), 3)
        y_pooled.append(y_te)
        fold_rows.append(row)
        print(f"  block {i}/{N_BLOCKS} done")

    y_all = np.concatenate(y_pooled)
    pooled = {name: np.vstack(parts) for name, parts in pooled.items()}

    print("\nLOG LOSS PER TEST BLOCK (lower is better)")
    print(f"{'block':<6}{'from':<12}{'to':<12}{'n':>4}" + "".join(f"{k:>11}" for k in NAMES))
    for r in fold_rows:
        print(f"{r['block']:<6}{r['from']:<12}{r['to']:<12}{r['test']:>4}"
              + "".join(f"{r[k]:>11.3f}" for k in NAMES))

    base_loss = per_match_loss(pooled["baseline"], y_all)
    summary = {}
    print(f"\nPOOLED OVER {len(y_all)} TEST MATCHES")
    print(f"{'model':<11}{'accuracy':>9}{'log loss':>10}{'vs baseline (log loss)':>26}   verdict")
    for name in NAMES:
        P = pooled[name]
        loss = per_match_loss(P, y_all)
        acc = float((P.argmax(axis=1) == y_all).mean())
        d = loss - base_loss
        diff = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        summary[name] = {
            "accuracy": round(acc, 3),
            "log_loss": round(float(loss.mean()), 3),
            "diff_vs_baseline": round(diff, 4),
            "diff_2se": round(2 * se, 4),
        }
        if name == "baseline":
            print(f"{name:<11}{acc:>9.3f}{loss.mean():>10.3f}{'-':>26}   -")
        else:
            summary[name]["verdict"] = verdict(diff, se)
            print(f"{name:<11}{acc:>9.3f}{loss.mean():>10.3f}"
                  f"{diff:>+16.3f} ± {2 * se:<6.3f}   {summary[name]['verdict']}")

    calib = {}
    for name in ["logit", "xgb", "blend"]:
        calib[name] = calibration(pooled[name], y_all)
        print(f"\nCALIBRATION: {name}")
        print(f"{'predicted range':<17}{'count':>7}{'avg predicted':>15}{'actually happened':>19}")
        for r in calib[name]:
            print(f"{r['range']:<17}{r['count']:>7}{r['predicted']:>15.3f}{r['observed']:>19.3f}")

    result = {
        "n_blocks": N_BLOCKS,
        "min_train": MIN_TRAIN,
        "test_matches": int(len(y_all)),
        "blocks": fold_rows,
        "summary": summary,
        "calibration": calib,
    }
    MODELS.mkdir(exist_ok=True)
    (MODELS / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nsaved models/evaluation.json")


if __name__ == "__main__":
    evaluate()