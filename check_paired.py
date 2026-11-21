import numpy as np
from pipeline.evaluate import FITTERS, MIN_TRAIN, N_BLOCKS, load_labeled, per_match_loss

labeled, features = load_labeled()
old_features = [f for f in features if "elo" not in f]
y = labeled["target"].to_numpy()
blocks = np.array_split(np.arange(MIN_TRAIN, len(labeled)), N_BLOCKS)
print("features: old", len(old_features), "| new", len(features))


def blend_losses(cols):
    X = labeled[cols].to_numpy()
    probs, ys = [], []
    for idx in blocks:
        start, stop = int(idx[0]), int(idx[-1]) + 1
        p1 = FITTERS["logit"](X[:start], y[:start], X[start:stop])
        p2 = FITTERS["xgb_small"](X[:start], y[:start], X[start:stop])
        probs.append((p1 + p2) / 2)
        ys.append(y[start:stop])
    return per_match_loss(np.vstack(probs), np.concatenate(ys))


old_loss = blend_losses(old_features)
new_loss = blend_losses(features)
d = new_loss - old_loss

se = d.std(ddof=1) / np.sqrt(len(d))
print(f"\nmatches: {len(d)}")
print(f"blend log loss: old {old_loss.mean():.3f} | new {new_loss.mean():.3f}")
print(f"paired difference (new - old): {d.mean():+.4f} +- {2 * se:.4f} (2 standard errors)")
print("verdict:", "better" if d.mean() < -2 * se else "worse" if d.mean() > 2 * se else "unclear")

start = 0
for i, idx in enumerate(blocks, start=1):
    block = d[start:start + len(idx)]
    start += len(idx)
    print(f"block {i}: {block.mean():+.3f}")