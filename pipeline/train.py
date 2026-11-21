import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from .config import MODELS
from .features import build_features, load_played
from .models import MODEL_NAME, blend_proba, fit_blend

TEST_SIZE = 380   # newest matches held back for a quick honest check


def score(y_true, probs):
    return {
        "accuracy": round(float(accuracy_score(y_true, probs.argmax(axis=1))), 3),
        "log_loss": round(float(log_loss(y_true, probs, labels=[0, 1, 2])), 3),
    }


def train():
    played = load_played()
    data, features = build_features(played)

    labeled = data.dropna(subset=["target"] + features).copy()
    labeled["target"] = labeled["target"].astype(int)
    labeled = labeled.sort_values("Date").reset_index(drop=True)

    train_set = labeled.iloc[:-TEST_SIZE]
    test_set = labeled.iloc[-TEST_SIZE:]
    X_train, y_train = train_set[features], train_set["target"]
    X_test, y_test = test_set[features], test_set["target"]
    print(f"training on {len(train_set)} matches, testing on {len(test_set)}")

    # quick check: learn from older matches, test on the newest 380
    rates = np.bincount(y_train, minlength=3) / len(y_train)
    base = score(y_test, np.tile(rates, (len(test_set), 1)))

    check = fit_blend(X_train, y_train)
    results = {
        "baseline (class rates)": base,
        "logistic regression": score(y_test, check["logit"].predict_proba(X_test)),
        "cautious XGBoost": score(y_test, check["xgb_small"].predict_proba(X_test)),
        "blend (saved model)": score(y_test, blend_proba(check, X_test)),
    }
    print(f"\n{'model':<26}{'accuracy':>10}{'log loss':>10}")
    for name, s in results.items():
        print(f"{name:<26}{s['accuracy']:>10}{s['log_loss']:>10}")
    blend_score = results["blend (saved model)"]

    # final model: learn from ALL finished matches, then save it
    final = fit_blend(labeled[features], labeled["target"])
    MODELS.mkdir(exist_ok=True)
    joblib.dump(
        {"members": final, "features": features, "name": MODEL_NAME},
        MODELS / "match_model.joblib",
    )

    metrics = {
        "model_name": MODEL_NAME,
        "test_period": f"{test_set['Date'].min():%d %b %Y} to {test_set['Date'].max():%d %b %Y}",
        "test_matches": int(len(test_set)),
        "accuracy": blend_score["accuracy"],
        "log_loss": blend_score["log_loss"],
        "baseline_accuracy": base["accuracy"],
        "baseline_log_loss": base["log_loss"],
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nmodel and metrics saved.")
    return metrics


if __name__ == "__main__":
    train()