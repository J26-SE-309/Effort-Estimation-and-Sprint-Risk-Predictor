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

from erp.arena import data
from erp.arena.predictor import Predictor, Run, StackPredictor, _logit, with_estimated_points
from erp.features import catalog
from erp.models import explain
from erp.models.inputs import design

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


def factors(model, stories: pd.DataFrame, text=None, run: Run | None = None,
            tasks: tuple[str, ...] = ("effort", "risk")) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """(effort contributions, risk contributions, method): one row per story, one column per planning factor.

    tasks: which to explain (TreeSHAP of a large effort model is the costliest step of a request, and the
    service shows only the risk reasons). run: a pass the caller already made, reused instead of predicting again.
    """
    how = method(model)
    if how == "TreeSHAP":
        return _tree(model, run or model.run(stories, text), tasks) + (how,)
    if how.startswith("TreeSHAP of the stack"):
        return _stack(model, stories, text, tasks) + (how,)
    return _occlusion(model, stories, text, tasks) + (how,)


def _tree(model: Predictor, run: Run, tasks=("effort", "risk")) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    out = []
    for task, learner in (("effort", model.effort_model), ("risk", model.risk_model)):
        out.append(explain.by_factor(learner.contributions(run.inputs[task]), catalog.factor_of)
                   if task in tasks else None)
    return tuple(out)


def _stack(model: StackPredictor, stories: pd.DataFrame, text, tasks=("effort", "risk")):
    parts = {}
    for name, base in model.bases.items():
        run = base.run(stories, None if text is None else text.get(base.encoder_name))
        parts[name] = _tree(base, run, tasks)
    manifest = model.manifest
    out = []
    for i, task in enumerate(("effort", "risk")):
        if task not in tasks:
            out.append(None)
            continue
        blend = sum(w * parts[n][i] for n, w in zip(manifest[task]["bases"], manifest[task]["meta"]["weights"],
                                                    strict=True))
        out.append(blend.fillna(0.0))
    return tuple(out)


def _baseline(model) -> dict[str, float]:
    """Training medians of the numeric features, from the learner's own TabularPrep."""
    prep = getattr(getattr(model, "effort_model", None), "prep", None)
    if prep is None and hasattr(model, "joint"):
        prep = model.joint.prep
    if prep is None and hasattr(model, "models"):  # DistilBERT
        prep = model.models["effort"].prep
    return dict(prep.medians) if prep is not None else {}


def _one_task(model, stories: pd.DataFrame, text, task: str) -> np.ndarray:
    """One task's raw output (log points, or the risk score) without running the other task's model."""
    if isinstance(model, Predictor) and not hasattr(model, "joint"):
        learner = model.effort_model if task == "effort" else model.risk_model
        return learner.predict(design(stories, text, task, text_columns=data.text_columns(model.encoder_name),
                                      levels=model.levels))
    if hasattr(model, "models"):  # DistilBERT
        return model.models[task].predict(text, design(stories, None, task, use_text=False, levels=model.levels))
    return model.raw(stories, text)[0 if task == "effort" else 1]  # M3 and stacks: one pass gives both


def _stacked(values: list):
    """Several copies of the text input as one batch (arrays, a stack's dict of arrays, or DistilBERT's texts)."""
    first = values[0]
    if isinstance(first, dict):
        return {name: np.vstack([v[name] for v in values]) for name in first}
    if isinstance(first, pd.Series):
        return pd.concat(values, ignore_index=True)
    return np.vstack(values)


def _occlusion(model, stories: pd.DataFrame, text, tasks=("effort", "risk")):
    """Each factor replaced by a neutral value in turn; all copies go through the model as one batch.

    M1's estimate is held fixed while the risk is occluded (M2 reads it for stories without points), so the
    risk reasons need only the risk model.
    """
    medians = _baseline(model)
    text = encode(model, stories) if text is None else text
    empty = encode(model, stories.assign(title="", description_text=""))
    log_points, raw = model.raw(stories, text)
    base = with_estimated_points(stories, log_points)
    factors = list(dict.fromkeys(catalog.factor_of(f.name) for f in catalog.FEATURES))
    copies, texts, which = [], [], []
    for number, factor in enumerate(factors):
        columns = [f.name for f in catalog.FEATURES if f.factor == factor]
        neutral = base.copy()
        for column in columns:
            neutral[column] = medians.get(column, np.nan)
        # a story already at the neutral values is unchanged: its contribution is exactly 0, no need to predict
        same = (base[columns] == neutral[columns]) | (base[columns].isna() & neutral[columns].isna())
        changed = np.ones(len(base), bool) if factor == catalog.TEXT_FACTOR else ~same.all(axis=1).to_numpy()
        if changed.any():
            copies.append(neutral[changed])
            factor_text = empty if factor == catalog.TEXT_FACTOR else text
            texts.append({k: v[changed] for k, v in factor_text.items()} if isinstance(factor_text, dict)
                         else factor_text[changed])
            which += [(number, row) for row in np.flatnonzero(changed)]
    out = []
    for task, original, transform in (("effort", log_points, np.asarray), ("risk", raw, _logit)):
        if task not in tasks:
            out.append(None)
            continue
        contribution = np.zeros((len(factors), len(stories)))
        if copies:
            occluded = np.asarray(_one_task(model, pd.concat(copies, ignore_index=True), _stacked(texts), task))
            for (number, row), value in zip(which, occluded, strict=True):
                contribution[number, row] = transform(original[row]) - transform(value)
        out.append(pd.DataFrame(contribution.T, index=stories.index, columns=factors))
    return tuple(out)


def reasons(contributions: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """Up to k factors pushing the prediction up, each with its share of the total push (FR14)."""
    return explain.top_reasons(contributions, k)
