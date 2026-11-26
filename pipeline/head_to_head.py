import numpy as np

from .evaluate import FITTERS, MIN_TRAIN, N_BLOCKS, load_labeled

NAMES = ["logit", "xgb_small", "blend"]
Z_FAMILY = 2.39   # about a 95% level across 3 comparisons (Bonferroni)


def per_match_loss(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1.0))


def collect():
    """Same expanding-window blocks as evaluate.py. Returns predictions, outcomes, block numbers."""
    labeled, features = load_labeled()
    X = labeled[features].to_numpy()
    y = labeled["target"].to_numpy()

    preds = {name: [] for name in NAMES}
    ys, blocks = [], []
    for b, idx in enumerate(np.array_split(np.arange(MIN_TRAIN, len(labeled)), N_BLOCKS), start=1):
        start, stop = int(idx[0]), int(idx[-1]) + 1
        X_tr, y_tr, X_te = X[:start], y[:start], X[start:stop]

        p_logit = FITTERS["logit"](X_tr, y_tr, X_te)
        p_xgb = FITTERS["xgb_small"](X_tr, y_tr, X_te)
        preds["logit"].append(p_logit)
        preds["xgb_small"].append(p_xgb)
        preds["blend"].append((p_logit + p_xgb) / 2)

        ys.append(y[start:stop])
        blocks.append(np.full(stop - start, b))

    preds = {name: np.vstack(parts) for name, parts in preds.items()}
    return preds, np.concatenate(ys), np.concatenate(blocks)


def compare(a, b, loss, blocks):
    """Paired difference: a minus b. Negative means a is better."""
    d = loss[a] - loss[b]
    diff = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    per_block = [float(d[blocks == k].mean()) for k in np.unique(blocks)]
    a_better = sum(1 for x in per_block if x < 0)
    return diff, se, per_block, a_better


def main():
    preds, y, blocks = collect()
    loss = {name: per_match_loss(P, y) for name, P in preds.items()}
    n_blocks = len(np.unique(blocks))

    print(f"{len(y)} out-of-sample matches, {n_blocks} blocks\n")
    print(f"{'model':<12}{'log loss':>10}{'accuracy':>10}")
    for name in NAMES:
        acc = float((preds[name].argmax(axis=1) == y).mean())
        print(f"{name:<12}{loss[name].mean():>10.4f}{acc:>10.3f}")

    print(f"\nPAIRED COMPARISONS (A minus B; negative = A is better)")
    print(f"{'A vs B':<22}{'diff':>9}{'±2 SE':>9}{'±2.39 SE':>10}  {'A better in':<14}verdict")
    for a, b in [("blend", "logit"), ("blend", "xgb_small"), ("logit", "xgb_small")]:
        diff, se, per_block, a_better = compare(a, b, loss, blocks)
        if diff < -Z_FAMILY * se:
            verdict = f"{a} better"
        elif diff > Z_FAMILY * se:
            verdict = f"{b} better"
        else:
            verdict = "unclear"
        print(f"{a + ' vs ' + b:<22}{diff:>+9.4f}{2 * se:>9.4f}{Z_FAMILY * se:>10.4f}  "
              f"{a_better} of {n_blocks:<9}{verdict}")
        print("   per block: " + "  ".join(f"{x:+.4f}" for x in per_block))

    avg_members = (loss["logit"].mean() + loss["xgb_small"].mean()) / 2
    print(f"\nEnsemble gain: the blend scores {avg_members - loss['blend'].mean():.4f} better "
          f"than the average of its two members")
    agree = float((preds["logit"].argmax(axis=1) == preds["xgb_small"].argmax(axis=1)).mean())
    print(f"The two members pick the same winner in {agree:.1%} of matches")
    corr = float(np.corrcoef(loss["logit"], loss["xgb_small"])[0, 1])
    print(f"Correlation of their per-match losses: {corr:.3f}")

    print(f"\nDECISION (rule fixed in advance): a challenger replaces the blend only if it beats "
          f"the blend by more than {Z_FAMILY} SE AND in at least {n_blocks - 1} of {n_blocks} blocks.")
    flagged = False
    for challenger in ["logit", "xgb_small"]:
        diff, se, _, better = compare(challenger, "blend", loss, blocks)
        if diff < -Z_FAMILY * se and better >= n_blocks - 1:
            print(f"-> {challenger} qualifies as a CANDIDATE (production stays unchanged until live rounds exist).")
            flagged = True
    if not flagged:
        print("-> No challenger qualifies. The blend stays.")


if __name__ == "__main__":
    main()