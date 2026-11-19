import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import MODELS
from .features import build_features, load_played

TEST_SIZE = 380   # test on the most recent ~1 season of matches


def make_xgb():
    return XGBClassifier(
        objective="multi:softprob",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def make_logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def score(y_true, probs):
    return {
        "accuracy": round(float(accuracy_score(y_true, probs.argmax(axis=1))), 3),
        "log_loss": round(float(log_loss(y_true, probs, labels=[0, 1, 2])), 3),
    }


def train():
    played = load_played()
    data, features = build_features(played)

    # keep only finished matches that have all their features
    labeled = data.dropna(subset=["target"] + features).copy()
    labeled["target"] = labeled["target"].astype(int)
    labeled = labeled.sort_values("Date").reset_index(drop=True)

    # split by TIME: old matches to learn from, newest matches to test on
    train_set = labeled.iloc[:-TEST_SIZE]
    test_set = labeled.iloc[-TEST_SIZE:]
    X_train, y_train = train_set[features], train_set["target"]
    X_test, y_test = test_set[features], test_set["target"]
    print(f"training on {len(train_set)} matches, testing on {len(test_set)}")

    # model 0: the dumb baseline (always predict the historical home/draw/away rates)
    rates = np.bincount(y_train, minlength=3) / len(y_train)
    base = score(y_test, np.tile(rates, (len(test_set), 1)))

    # model 1: logistic regression
    logit = make_logit().fit(X_train, y_train)
    logit_score = score(y_test, logit.predict_proba(X_test))

    # model 2: XGBoost
    xgb = make_xgb().fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)
    xgb_score = score(y_test, xgb_probs)

    print(f"\n{'model':<26}{'accuracy':>10}{'log loss':>10}")
    for name, s in [("baseline (class rates)", base),
                    ("logistic regression", logit_score),
                    ("XGBoost", xgb_score)]:
        print(f"{name:<26}{s['accuracy']:>10}{s['log_loss']:>10}")

    picks = np.bincount(xgb_probs.argmax(axis=1), minlength=3)
    print(f"\nXGBoost picked -> home: {picks[0]}, draw: {picks[1]}, away: {picks[2]}")

    # final model: learn from ALL finished matches, then save it
    final = make_xgb().fit(labeled[features], labeled["target"])
    ranked = sorted(zip(final.feature_importances_, features), reverse=True)[:8]
    print("\nfeatures the model leans on most:")
    for importance, name in ranked:
        print(f"  {name:<24}{importance:.3f}")

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": features}, MODELS / "xgb_match.joblib")

    metrics = {
        "test_period": f"{test_set['Date'].min():%d %b %Y} to {test_set['Date'].max():%d %b %Y}",
        "test_matches": int(len(test_set)),
        "accuracy": xgb_score["accuracy"],
        "log_loss": xgb_score["log_loss"],
        "baseline_accuracy": base["accuracy"],
        "baseline_log_loss": base["log_loss"],
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nmodel and metrics saved.")
    return metrics


if __name__ == "__main__":
    train()