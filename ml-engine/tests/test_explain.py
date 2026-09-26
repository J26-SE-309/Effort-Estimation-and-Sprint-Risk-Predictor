"""The planted-signal test (ML guide 7.2): a debugging check of the pipeline, never a research finding.

Made-up stories where only two clues decide the risk: a story is usually at risk when its wording is vague
AND it has an open blocker. The real training code (first_models.fit) and explanation code (explain) must find
exactly that: the two planted clues on top, the noise clue behind both, and a planted clue as the first
reason for (nearly) every risky story. If this fails, there is a bug in the pipeline, not a finding.

Observed with the untuned Phase 2 settings: the noise clue still gets about a quarter of the mean absolute
SHAP value, because the trees also fit the random part of the labels. SHAP reports that faithfully; it is a
model issue for the Phase 4 tuning (stronger regularisation took it to 16% here), not an explanation bug.
"""

import numpy as np
import pandas as pd

from erp.models import explain, first_models


def planted_data(n: int = 4000, seed: int = 42) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({
        "ambiguity_score": rng.uniform(0, 1, n),
        "blocker_count": rng.poisson(0.5, n),
        "noise_feature": rng.normal(0, 1, n),  # must come out as unimportant
    })
    p = np.where((x["ambiguity_score"] > 0.7) & (x["blocker_count"] > 0), 0.8, 0.1)
    y = pd.Series((rng.random(n) < p).astype(int))
    parts = pd.Series(np.where(np.arange(n) < int(n * 0.75), "train", "cal"))
    return x, y, parts


def test_planted_signal_is_found_and_explained():
    x, y, parts = planted_data()
    model = first_models.fit("risk", x, y, parts)
    shap = explain.contributions(model.booster_, x)
    importance = explain.global_importance(shap)
    assert set(importance.index[:2]) == {"ambiguity_score", "blocker_count"}
    assert importance["noise_feature"] < min(importance["ambiguity_score"], importance["blocker_count"])

    planted = (x["ambiguity_score"] > 0.7) & (x["blocker_count"] > 0)
    first = [row[0]["factor"] if row else None for row in explain.top_reasons(shap.loc[planted])]
    assert sum(f in {"ambiguity_score", "blocker_count"} for f in first) / len(first) >= 0.98
