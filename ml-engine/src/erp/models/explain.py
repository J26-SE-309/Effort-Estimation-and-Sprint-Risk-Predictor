"""X1: explain each prediction as contributions of planning factors, and keep the top three reasons.

LightGBM computes exact TreeSHAP values itself (predict with pred_contrib=True), so no extra library is
needed. The values of all features of one planning factor are added up; the 384 SBERT values become the single
factor "Story size and content", because one embedding dimension means nothing to a user (ML guide 4.7).
For M2 the values are in log-odds, before calibration: fine for ranking reasons, not to be read as
percentage points.
"""

import numpy as np
import pandas as pd


def contributions(booster, x: pd.DataFrame) -> pd.DataFrame:
    """SHAP value of every input column for every row (the last column, the base value, is dropped)."""
    values = booster.predict(x, pred_contrib=True)
    return pd.DataFrame(np.asarray(values)[:, :-1], index=x.index, columns=x.columns)


def by_factor(shap: pd.DataFrame, factor_of) -> pd.DataFrame:
    """Add up the SHAP values of the columns that belong to the same planning factor."""
    return shap.T.groupby(lambda column: factor_of(column)).sum().T


MIN_REASON_SHARE = 0.05  # a factor behind less than 5% of the total push is noise, not a reason


def top_reasons(factors: pd.DataFrame, k: int = 3, min_share: float = MIN_REASON_SHARE) -> list[list[dict]]:
    """Per row, up to k factors pushing the prediction up the most (towards more effort or more risk).

    share is the factor's part of the total absolute contribution; factors below min_share are left out.
    """
    reasons = []
    for _, row in factors.iterrows():
        total = row.abs().sum() or 1.0
        up = (row[row > 0] / total).sort_values(ascending=False)
        up = up[up >= min_share].head(k)
        reasons.append([{"factor": factor, "share": round(float(share), 3)} for factor, share in up.items()])
    return reasons


def global_importance(factors: pd.DataFrame) -> pd.Series:
    """Mean absolute contribution per factor, as a share of the total (for the report)."""
    mean = factors.abs().mean()
    return (mean / mean.sum()).sort_values(ascending=False)
