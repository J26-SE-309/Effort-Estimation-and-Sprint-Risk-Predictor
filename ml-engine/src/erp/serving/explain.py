"""X1 in the service: every prediction explained as contributions of planning factors (ML guide 4.7, FR14).

Three ways, depending on the configuration, all giving the same thing: for each story and planning factor, how
far it pushed the prediction (log points for effort, log-odds for risk):
  TreeSHAP   LightGBM, XGBoost and CatBoost compute exact SHAP values themselves.
  stack      its meta-models are linear in the members' outputs (log points; risk logits), so the members'
             SHAP values, weighted by the meta weights, are exact SHAP values of the stack.
  occlusion  Random Forest, SVR / SVM and the networks: each factor in turn is replaced by a neutral value (the
             training median of its features; for the story text, an empty story) and the change in the
             prediction is that factor's contribution. It needs one extra prediction per factor, no extra library.
The contributions of all columns of one factor are added up (the 384 SBERT values become "Story size and
content"); the reasons are the factors pushing the risk up the most (explain.top_reasons).
"""

import numpy as np
import pandas as pd

from erp.arena.predictor import Predictor, Run, StackPredictor, _logit
from erp.features import catalog
from erp.models import explain

TREE_LEARNERS = {"lightgbm", "xgboost", "catboost"}


def encode(model, stories: pd.DataFrame):
    """The text input for a configuration, encoded from scratch as for a story never seen before (no cache: the
    service keeps no files). A stack gets one matrix per encoder its members use."""
    if isinstance(model, StackPredictor):
        by_encoder = {base.encoder_name: base for base in model.bases.values()}
        return {name: base.encode(stories, fresh=True) for name, base in by_encoder.items()}
    return model.encode(stories, fresh=True)


def method(model) -> str:
    learner = model.manifest["config"]["learner"]
    if learner in TREE_LEARNERS:
        return "TreeSHAP"
    if isinstance(model, StackPredictor) and all(
            b.manifest["config"]["learner"] in TREE_LEARNERS for b in model.bases.values()):
        return "TreeSHAP of the stack's members, weighted by its meta-models"
    return "feature-group occlusion"


def factors(model, stories: pd.DataFrame, text=None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(effort contributions, risk contributions, method): one row per story, one column per planning factor."""
    how = method(model)
    if how == "TreeSHAP":
        run = model.run(stories, text)
        return _tree(model, run) + (how,)
    if how.startswith("TreeSHAP of the stack"):
        return _stack(model, stories, text) + (how,)
    return _occlusion(model, stories, text) + (how,)


def _tree(model: Predictor, run: Run) -> tuple[pd.DataFrame, pd.DataFrame]:
    effort = model.effort_model.contributions(run.inputs["effort"])
    risk = model.risk_model.contributions(run.inputs["risk"])
    return explain.by_factor(effort, catalog.factor_of), explain.by_factor(risk, catalog.factor_of)


def _stack(model: StackPredictor, stories: pd.DataFrame, text) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = {}
    for name, base in model.bases.items():
        run = base.run(stories, None if text is None else text.get(base.encoder_name))
        parts[name] = _tree(base, run)
    manifest = model.manifest
    effort = sum(w * parts[n][0] for n, w in zip(manifest["effort"]["bases"], manifest["effort"]["meta"]["weights"],
                                                   strict=True))
    risk = sum(w * parts[n][1] for n, w in zip(manifest["risk"]["bases"], manifest["risk"]["meta"]["weights"],
                                                 strict=True))
    return effort.fillna(0.0), risk.fillna(0.0)


def _baseline(model) -> dict[str, float]:
    """Training medians of the numeric features, from the learner's own TabularPrep."""
    prep = getattr(getattr(model, "effort_model", None), "prep", None)
    if prep is None and hasattr(model, "joint"):
        prep = model.joint.prep
    if prep is None and hasattr(model, "models"):  # DistilBERT
        prep = model.models["effort"].prep
    return dict(prep.medians) if prep is not None else {}


def _occlusion(model, stories: pd.DataFrame, text) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = _baseline(model)
    text = encode(model, stories) if text is None else text
    empty = encode(model, stories.assign(title="", description_text=""))
    log_points, raw = model.raw(stories, text)
    by_factor = {}
    for factor in dict.fromkeys(catalog.factor_of(f.name) for f in catalog.FEATURES):
        columns = [f.name for f in catalog.FEATURES if f.factor == factor]
        neutral = stories.copy()
        for column in columns:
            neutral[column] = medians.get(column, np.nan)
        occluded_text = empty if factor == catalog.TEXT_FACTOR else text
        occluded_log, occluded_raw = model.raw(neutral, occluded_text)
        by_factor[factor] = (log_points - occluded_log, _logit(raw) - _logit(occluded_raw))
    effort = pd.DataFrame({f: v[0] for f, v in by_factor.items()}, index=stories.index)
    risk = pd.DataFrame({f: v[1] for f, v in by_factor.items()}, index=stories.index)
    return effort, risk


def reasons(contributions: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """Up to k factors pushing the prediction up, each with its share of the total push (FR14)."""
    return explain.top_reasons(contributions, k)
