from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_NAME = "Blend + Elo (logistic regression + cautious XGBoost)"


def make_xgb():
    """The original XGBoost. Kept only for comparison in evaluate.py."""
    return XGBClassifier(
        objective="multi:softprob",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def make_xgb_small():
    """A more cautious XGBoost: shallower, fewer trees, more data needed per leaf."""
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


def make_logit():
    return make_pipeline(
        SimpleImputer(strategy="median"),   # fills missing values (see below)
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def fit_blend(X, y):
    """Train both members of the blend. Returns a plain dict, which is safe to save."""
    return {
        "logit": make_logit().fit(X, y),
        "xgb_small": make_xgb_small().fit(X, y),
    }


def blend_proba(members, X):
    """Average the two models' probabilities."""
    return (members["logit"].predict_proba(X) + members["xgb_small"].predict_proba(X)) / 2