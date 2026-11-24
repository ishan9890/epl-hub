import numpy as np
from scipy.optimize import minimize_scalar

from .evaluate import FITTERS, MIN_TRAIN, N_BLOCKS, load_labeled

EDGES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0001]


def blend_blocks():
    """Same expanding-window blocks as evaluate.py: out-of-sample blend predictions per block."""
    labeled, features = load_labeled()
    X = labeled[features].to_numpy()
    y = labeled["target"].to_numpy()

    blocks = []
    for i, idx in enumerate(np.array_split(np.arange(MIN_TRAIN, len(labeled)), N_BLOCKS), start=1):
        start, stop = int(idx[0]), int(idx[-1]) + 1
        X_tr, y_tr, X_te = X[:start], y[:start], X[start:stop]
        P = (FITTERS["logit"](X_tr, y_tr, X_te) + FITTERS["xgb_small"](X_tr, y_tr, X_te)) / 2
        blocks.append({"block": i, "P": P, "y": y[start:stop]})
    return blocks


def per_match_loss(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1.0))


def mean_log_loss(P, y):
    return float(per_match_loss(P, y).mean())


def apply_temperature(P, T):
    logp = np.log(np.clip(P, 1e-9, 1.0)) / T
    logp -= logp.max(axis=1, keepdims=True)
    Q = np.exp(logp)
    return Q / Q.sum(axis=1, keepdims=True)


def fit_temperature(P, y):
    result = minimize_scalar(
        lambda T: mean_log_loss(apply_temperature(P, T), y),
        bounds=(0.5, 3.0),
        method="bounded",
    )
    return float(result.x)


def walk_forward(blocks):
    """For each block after the first, fit T on all EARLIER blocks, apply it to this block."""
    rows, raw_parts, cal_parts, y_parts = [], [], [], []
    for i in range(1, len(blocks)):
        P_past = np.vstack([b["P"] for b in blocks[:i]])
        y_past = np.concatenate([b["y"] for b in blocks[:i]])
        T = fit_temperature(P_past, y_past)

        P, y = blocks[i]["P"], blocks[i]["y"]
        Q = apply_temperature(P, T)
        rows.append((blocks[i]["block"], T, mean_log_loss(P, y), mean_log_loss(Q, y)))
        raw_parts.append(P)
        cal_parts.append(Q)
        y_parts.append(y)
    return rows, np.vstack(raw_parts), np.vstack(cal_parts), np.concatenate(y_parts)


def reliability(P, y):
    p = P.ravel()
    outcome = np.eye(3)[y].ravel()
    bucket = np.digitize(p, EDGES) - 1
    rows = []
    for b in range(len(EDGES) - 1):
        m = bucket == b
        if m.sum() >= 20:
            rows.append((f"{EDGES[b]:.1f}-{min(EDGES[b + 1], 1.0):.1f}",
                         int(m.sum()), float(p[m].mean()), float(outcome[m].mean())))
    return rows


def print_reliability(title, P, y):
    rows = reliability(P, y)
    print(f"\n{title}")
    print(f"{'range':<10}{'count':>7}{'avg predicted':>15}{'happened':>10}{'gap':>8}")
    for rng, n, pred, obs in rows:
        print(f"{rng:<10}{n:>7}{pred:>15.3f}{obs:>10.3f}{obs - pred:>+8.3f}")
    avg_gap = sum(n * abs(pred - obs) for _, n, pred, obs in rows) / sum(n for _, n, _, _ in rows)
    print(f"weighted average absolute gap: {avg_gap:.4f}")


def class_check(P, y):
    print(f"\n{'outcome':<8}{'avg predicted':>15}{'happened':>10}{'±2 SE':>8}")
    for k, name in enumerate(["home", "draw", "away"]):
        obs = float((y == k).mean())
        two_se = 2 * np.sqrt(obs * (1 - obs) / len(y))
        print(f"{name:<8}{P[:, k].mean():>15.3f}{obs:>10.3f}{two_se:>8.3f}")


def main():
    blocks = blend_blocks()

    print("RAW BLEND, PER BLOCK (should match the blend column of evaluate.py)")
    for b in blocks:
        print(f"  block {b['block']}: {len(b['y'])} matches, log loss {mean_log_loss(b['P'], b['y']):.3f}")

    rows, P, Q, y = walk_forward(blocks)

    print(f"\nWALK-FORWARD TEMPERATURE SCALING, blocks 2-{len(blocks)}")
    print(f"{'block':<8}{'T (fit on earlier blocks)':>27}{'raw':>9}{'calibrated':>12}")
    for block, T, raw, cal in rows:
        print(f"{block:<8}{T:>27.3f}{raw:>9.3f}{cal:>12.3f}")

    d = per_match_loss(Q, y) - per_match_loss(P, y)
    diff = float(d.mean())
    two_se = float(2 * d.std(ddof=1) / np.sqrt(len(d)))
    better = sum(1 for _, _, raw, cal in rows if cal < raw)

    print(f"\nPOOLED OVER {len(y)} MATCHES")
    print(f"raw log loss        {mean_log_loss(P, y):.4f}")
    print(f"calibrated log loss {mean_log_loss(Q, y):.4f}")
    print(f"paired difference   {diff:+.4f} ± {two_se:.4f}   (negative = calibration helps)")
    print(f"blocks improved     {better} of {len(rows)}")
    same_pick = float((P.argmax(axis=1) == Q.argmax(axis=1)).mean())
    print(f"predicted winner unchanged in {same_pick:.1%} of matches (should be 100%)")

    print_reliability("RELIABILITY BEFORE", P, y)
    print_reliability("RELIABILITY AFTER", Q, y)
    class_check(P, y)

    print("\nDECISION (rule fixed in advance): adopt only if the improvement beats 2 SE "
          f"AND calibration helped in at least {len(rows) - 1} of {len(rows)} blocks.")
    if diff < -two_se and better >= len(rows) - 1:
        print("-> ADOPT temperature scaling.")
    else:
        print("-> DO NOT ADOPT. Record it as a finding. Do not try other calibrators on this data.")


if __name__ == "__main__":
    main()