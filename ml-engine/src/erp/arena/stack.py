"""The stacked ensemble (Appendix D, configuration 9): the best few single-task configurations, combined.

Which bases: for each task, the STACK_SIZE configurations with the best inner-fold score from their own tuning,
so the choice never looks at the calibration or test split. How they are combined is learned from their
inner-fold predictions (each story predicted by a model that never saw it, trained only on earlier sprints),
not from the calibration split, which stays reserved for C1, C2 and the threshold:
  effort  a linear blend of the bases' log(1 + points) predictions with non-negative weights
  risk    a logistic regression on the logits of the bases' risk scores
Both are a handful of numbers, saved in model.json; the bases stay in their own folders and are reloaded
from there.
"""

import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from erp.arena import configs, predictor
from erp.arena.data import MODELS_DIR, WORK_DIR, ArenaData, targets
from erp.arena.train import directory_size, log, uncertainty
from erp.models.bundle import RISK_BANDS, code_version


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def choose_bases() -> dict[str, list[str]]:
    scores = {}
    for name in configs.STACK_BASES:
        path = MODELS_DIR / name / predictor.MANIFEST
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            scores[name] = {task: manifest[task]["tuning"]["best_value"] for task in ("effort", "risk")}
    return {"effort": sorted(scores, key=lambda n: scores[n]["effort"])[: configs.STACK_SIZE],
            "risk": sorted(scores, key=lambda n: -scores[n]["risk"])[: configs.STACK_SIZE],
            "scores": scores}


def build(arena: ArenaData, force: bool = False) -> bool:
    """Fits and saves the stack; False when fewer than two bases are trained yet."""
    from sklearn.linear_model import LinearRegression, LogisticRegression

    directory = MODELS_DIR / "stack"
    if (directory / predictor.MANIFEST).exists() and not force:
        return True
    chosen = choose_bases()
    if len(chosen["scores"]) < 2:
        return False
    config = configs.BY_ID["stack"]
    names = sorted(set(chosen["effort"]) | set(chosen["risk"]))
    oof = {n: pd.read_parquet(WORK_DIR / "oof" / f"{n}.parquet") for n in names}
    rows = oof[names[0]].index
    y_effort = targets(arena.frame.loc[rows], "effort")
    y_risk = targets(arena.frame.loc[rows], "risk")

    effort_meta = LinearRegression(positive=True).fit(
        np.column_stack([oof[n].loc[rows, "effort_log"] for n in chosen["effort"]]), y_effort)
    risk_meta = LogisticRegression(C=1.0).fit(
        np.column_stack([_logit(oof[n].loc[rows, "risk_raw"]) for n in chosen["risk"]]), y_risk)

    manifest = {"config": asdict(config), "arena": configs.ARENA, "code": code_version(),
                "encoder": {"name": "several", "bases": "each base uses its own encoder"},
                "data": {"source": "TAWOS v1.1", "stories": len(arena.frame),
                         "split": arena.parts.value_counts().to_dict()}}
    manifest["effort"] = {"bases": chosen["effort"], "meta": {
        "model": "linear blend of log(1 + points), non-negative weights",
        "weights": effort_meta.coef_.round(6).tolist(), "intercept": round(float(effort_meta.intercept_), 6)}}
    manifest["risk"] = {"bases": chosen["risk"], "meta": {
        "model": "logistic regression on the logits of the bases' scores",
        "weights": risk_meta.coef_[0].round(6).tolist(), "intercept": round(float(risk_meta.intercept_[0]), 6)}}
    manifest["selection"] = {
        "rule": f"the {configs.STACK_SIZE} best configurations per task by their inner-fold tuning score",
        "inner_fold_scores": chosen["scores"],
        "meta_fitted_on": f"{len(rows):,} inner-fold predictions (training split only)"}

    predictions = {n: pd.read_parquet(WORK_DIR / "predictions" / f"{n}.parquet") for n in names}
    outputs = {n: (predictions[n]["effort_log"].to_numpy(), predictions[n]["risk_raw"].to_numpy()) for n in names}
    log_points, raw = predictor.combine(manifest, outputs)
    fitted = uncertainty(arena, log_points, raw)
    manifest["effort"]["intervals"] = fitted["intervals"].to_dict()
    manifest["risk"].update({"calibrator": fitted["calibrator"].to_dict(), "threshold": fitted["threshold"],
                             "bands": dict(RISK_BANDS)})
    manifest["created"] = datetime.now(UTC).isoformat(timespec="seconds")

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    path = directory / predictor.MANIFEST
    path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    pd.DataFrame({"split": arena.parts, "effort_log": log_points, "risk_raw": raw},
                 index=arena.frame.index).to_parquet(WORK_DIR / "predictions" / "stack.parquet")

    loaded = predictor.StackPredictor(directory)
    text = {base.encoder_name: arena.full_text(base.encoder_name) for base in loaded.bases.values()}
    again_log, again_raw = loaded.raw(arena.frame, text)
    if not (np.array_equal(again_log, log_points) and np.array_equal(again_raw, raw)):
        raise AssertionError("stack: the reloaded ensemble does not reproduce its predictions")
    manifest["check"] = {"round_trip": "identical predictions for all stories after reloading from disk",
                         "size_mb": directory_size(directory)}
    path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    log(f"stack: effort from {', '.join(chosen['effort'])}; risk from {', '.join(chosen['risk'])}")
    from erp.arena import intervals  # adaptive C2 and the confidence score

    intervals.fit(arena, "stack")
    return True
