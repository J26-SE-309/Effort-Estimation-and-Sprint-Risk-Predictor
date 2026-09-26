"""Evaluation metrics, computed the way ML guide section 9 requires.

Effort metrics are per issue, on each project's own story-point scale. Standardised Accuracy compares a
model with random guessing (Shepperd and MacDonell, 2012): each guess is the actual story points of another,
randomly drawn issue of the same project in the evaluated set, averaged over many runs. Risk metrics are for
the at-risk class; the decision threshold is chosen on the calibration split, never on the test split.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

MIN_RECALL = 0.70  # NFR: recall of at-risk stories at least 0.70


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def mdae(y_true, y_pred) -> float:
    return float(np.median(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def random_guessing_mae(y_true, groups, runs: int = 1000, seed: int = 42) -> float:
    """Mean MAE of random guessing: each issue gets the actual value of another issue of its group."""
    y = np.asarray(y_true, float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    errors = np.zeros(runs)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if len(idx) < 2:
            continue
        # draw another member: an offset in 1..n-1 from the issue's own position, wrapped around
        offsets = rng.integers(1, len(idx), size=(runs, len(idx)))
        guesses = y[idx][(np.arange(len(idx)) + offsets) % len(idx)]
        errors += np.abs(guesses - y[idx]).sum(axis=1)
    return float(errors.mean() / len(y))


def standardised_accuracy(y_true, y_pred, groups, runs: int = 1000, seed: int = 42) -> float:
    """SA = (1 - MAE_model / MAE_random) x 100. Above 0 beats random guessing; negative is worse."""
    return (1 - mae(y_true, y_pred) / random_guessing_mae(y_true, groups, runs, seed)) * 100


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    """ECE with equal-width bins: sum over bins of (bin share) x |mean predicted - observed rate|."""
    y, p = np.asarray(y_true, float), np.asarray(probabilities, float)
    which = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def choose_threshold(y_true, probabilities, min_recall: float = MIN_RECALL) -> float:
    """The threshold with the best F1 among those keeping recall >= min_recall (on the calibration split)."""
    y, p = np.asarray(y_true, bool), np.asarray(probabilities, float)
    best, best_f1 = 0.5, -1.0
    for threshold in np.unique(np.round(p, 3)):
        flagged = p >= threshold
        tp = (flagged & y).sum()
        recall = tp / max(y.sum(), 1)
        if recall < min_recall:
            continue
        precision = tp / max(flagged.sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best, best_f1 = float(threshold), f1
    return best


def classification_report(y_true, probabilities, threshold: float) -> dict:
    y, p = np.asarray(y_true, bool), np.asarray(probabilities, float)
    flagged = p >= threshold
    tp = (flagged & y).sum()
    precision = tp / max(flagged.sum(), 1)
    recall = tp / max(y.sum(), 1)
    single_class = len(np.unique(y)) < 2
    return {
        "precision": float(precision), "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y, p)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
    }


def effort_report(y_true, y_pred, groups) -> dict:
    return {"mae": mae(y_true, y_pred), "mdae": mdae(y_true, y_pred),
            "sa": standardised_accuracy(y_true, y_pred, groups)}


def per_group(frame: pd.DataFrame, group: str, metric) -> pd.Series:
    return frame.groupby(group).apply(metric, include_groups=False)


def wilcoxon_p(errors_a, errors_b) -> float:
    """Two-sided Wilcoxon signed-rank test on paired absolute errors (ML guide section 9)."""
    from scipy.stats import wilcoxon

    return float(wilcoxon(np.asarray(errors_a, float), np.asarray(errors_b, float)).pvalue)


def auc_difference(y_true, scores_a, scores_b, runs: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """ROC-AUC of a minus ROC-AUC of b, with a 95% paired bootstrap interval over the evaluated stories."""
    y, a, b = np.asarray(y_true, bool), np.asarray(scores_a, float), np.asarray(scores_b, float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(runs):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].all() or not y[idx].any():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(roc_auc_score(y, a) - roc_auc_score(y, b)), float(low), float(high)


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, for comparing many configurations at once (ML guide 9 and 10.1)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def vargha_delaney(a, b) -> float:
    """Vargha-Delaney A12: the chance that a value from a is larger than one from b (ties count half).

    0.5 means no difference; for errors, A12 below 0.5 means a tends to have the smaller errors.
    Conventional thresholds: 0.56 small, 0.64 medium, 0.71 large (and mirrored below 0.5).
    """
    from scipy.stats import rankdata

    a, b = np.asarray(a, float), np.asarray(b, float)
    ranks = rankdata(np.concatenate([a, b]))
    return float((ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def effect_size_label(a12: float) -> str:
    distance = abs(a12 - 0.5)
    return "negligible" if distance < 0.06 else "small" if distance < 0.14 else "medium" if distance < 0.21 \
        else "large"


def paired_bootstrap(metric, y_true, scores_a, scores_b, runs: int = 1000, seed: int = 42) -> dict:
    """metric(a) - metric(b) on the same stories, with a 95% interval and a two-sided bootstrap p-value.

    metric(y_true, scores) -> float. Resamples stories (with replacement), keeping each story's pair together.
    """
    y, a, b = np.asarray(y_true), np.asarray(scores_a, float), np.asarray(scores_b, float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(runs):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2 and y.dtype == bool:
            continue
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    diffs = np.asarray(diffs)
    low, high = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"difference": float(metric(y, a) - metric(y, b)), "low": float(low), "high": float(high),
            "p": float(max(p, 1 / len(diffs)))}
