"""Phase 4: tune, train and save every arena configuration (ML guide section 8, step 7).

For each configuration of configs.CONFIGS (DistilBERT needs a GPU and has its own script; the stack is built
from the others by erp-arena-report):
  1. Tune with Optuna on the inner folds of the training split: the same N_TRIALS trials for every
     configuration, a TPE sampler with a fixed seed, and a median pruner that stops a trial early when it is
     clearly worse than the others after the first folds. Effort is tuned for MAE in story points, risk for
     ROC-AUC (the threshold and calibration come later), M3 for its validation loss.
  2. Train the best settings on the whole training split. Boosters and networks stop early on the calibration
     split, as in Phases 2 and 3.
  3. Fit the C2 intervals (effort) and the C1 calibrator and the risk threshold on the calibration split.
  4. Save the configuration to ml-engine/models/arena-v1/<id>/, reload it from disk and check that it
     reproduces every prediction exactly. The 50-story latency (NFR1) is timed by erp-arena-report, for all
     configurations one after another on an otherwise idle machine, so parallel training runs cannot skew it.
The predictions, the inner-fold predictions (for the stack) and every Optuna trial go to
Datasets/effort-risk/arena/arena-v1/, because they are data about TAWOS stories.

Usage:
    erp-train-arena                          # every configuration that is not saved yet
    erp-train-arena --configs tfidf-svm      # only these
    erp-train-arena --prepare                # only fit and cache the text encoders (do this before parallel runs)
    erp-train-arena --force                  # retrain even if already saved
"""

import argparse
import json
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from erp.arena import configs, data, predictor
from erp.arena.configs import ARENA, SEED, Config
from erp.arena.data import MODELS_DIR, WORK_DIR, ArenaData, targets, text_columns
from erp.models import calibration, metrics
from erp.models.bundle import INTERVAL_COVERAGES, RISK_BANDS, code_version
from erp.models.encoders import SbertEncoder
from erp.models.inputs import design
from erp.models.learners import LEARNERS
from erp.models.mlp import MLP

LATENCY_STORIES = 50
LATENCY_RUNS = 20


def log(message: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def fold_score(task: str, y_true: np.ndarray, predicted: np.ndarray) -> float:
    if task == "effort":
        return metrics.mae(np.expm1(y_true), np.expm1(predicted))
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, predicted))


def _study(direction: str):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=SEED),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=5))


def _progress(name: str, trials: int, started: float):
    def callback(study, trial):
        best = study.best_value if any(t.value is not None for t in study.trials) else float("nan")
        value = "pruned" if trial.value is None else f"{trial.value:.4f}"
        log(f"  {name} trial {trial.number + 1}/{trials}: {value} (best {best:.4f}, "
            f"{(time.perf_counter() - started) / 60:.1f} min)")
    return callback


def _tuning_summary(study, metric: str, started: float) -> dict:
    import optuna

    states = [t.state for t in study.trials]
    return {"trials": len(states), "complete": states.count(optuna.trial.TrialState.COMPLETE),
            "pruned": states.count(optuna.trial.TrialState.PRUNED), "metric": metric,
            "inner_folds": configs.INNER_FOLDS, "sampler": f"TPE (seed {SEED}), median pruner",
            "best_value": float(study.best_value), "best_params": study.best_params,
            "seconds": round(time.perf_counter() - started, 1)}


def tune_single(config: Config, task: str, arena: ArenaData, trials: int):
    """Returns (study, inner-fold predictions of the best trial, tuning summary)."""
    import optuna

    learner = LEARNERS[config.learner]
    train, columns = arena.train, text_columns(config.encoder)
    y = targets(train, task)
    folds = []
    for i, (fit, val) in enumerate(arena.folds):
        x = design(train, arena.fold_text(config.encoder, i), task, text_columns=columns, levels=arena.levels)
        folds.append((x[fit], y[fit], x[val], y[val]))
    predictions = {}

    def objective(trial):
        params = learner(task).space(trial)
        scores, sizes, fold_predictions = [], [], []
        for i, (x_fit, y_fit, x_val, y_val) in enumerate(folds):
            predicted = learner(task).fit(x_fit, y_fit, x_val, y_val, params).predict(x_val)
            fold_predictions.append(predicted)
            scores.append(fold_score(task, y_val, predicted))
            sizes.append(len(y_val))
            trial.report(float(np.average(scores, weights=sizes)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        predictions[trial.number] = fold_predictions
        return float(np.average(scores, weights=sizes))

    started = time.perf_counter()
    study = _study("minimize" if task == "effort" else "maximize")
    study.optimize(objective, n_trials=trials, callbacks=[_progress(f"{config.id} {task}", trials, started)])
    metric = "MAE in story points" if task == "effort" else "ROC-AUC"
    return study, predictions[study.best_trial.number], _tuning_summary(study, metric, started)


def tune_joint(config: Config, arena: ArenaData, trials: int, tasks=("effort", "risk")):
    """M3 (or a single-task twin, for H1): tuned for the validation loss of its tasks."""
    import optuna

    train, columns = arena.train, text_columns(config.encoder)
    y = {task: targets(train, task) for task in tasks}
    folds = []
    for i, (fit, val) in enumerate(arena.folds):
        x = design(train, arena.fold_text(config.encoder, i), "joint", text_columns=columns, levels=arena.levels)
        folds.append((x[fit], {t: v[fit] for t, v in y.items()}, x[val], {t: v[val] for t, v in y.items()}))
    predictions = {}

    def objective(trial):
        params = MLP.space(trial)
        losses, sizes, fold_predictions = [], [], []
        for i, (x_fit, y_fit, x_val, y_val) in enumerate(folds):
            model = MLP(tasks).fit(x_fit, y_fit, x_val, y_val, params)
            fold_predictions.append(model.predict(x_val))
            losses.append(model.best_val_loss)
            sizes.append(len(x_val))
            trial.report(float(np.average(losses, weights=sizes)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        predictions[trial.number] = fold_predictions
        return float(np.average(losses, weights=sizes))

    started = time.perf_counter()
    study = _study("minimize")
    study.optimize(objective, n_trials=trials, callbacks=[_progress(f"{config.id} {'+'.join(tasks)}", trials,
                                                                    started)])
    metric = "validation loss (" + " + ".join({"effort": "Huber", "risk": "cross-entropy"}[t] for t in tasks) + ")"
    return study, predictions[study.best_trial.number], _tuning_summary(study, metric, started)


def uncertainty(arena: ArenaData, log_points: np.ndarray, raw: np.ndarray) -> dict:
    """C2 on the calibration split for effort; C1 and the threshold on it for risk."""
    cal = arena.mask("cal")
    y_effort, y_risk = targets(arena.frame, "effort"), targets(arena.frame, "risk")
    intervals = calibration.ConformalIntervals().fit(y_effort[cal], log_points[cal], INTERVAL_COVERAGES)
    calibrator = calibration.Calibrator("auto").fit(raw[cal], y_risk[cal])
    threshold = metrics.choose_threshold(y_risk[cal], calibrator.predict(raw[cal]))
    return {"intervals": intervals, "calibrator": calibrator, "threshold": threshold}


def directory_size(directory: Path) -> float:
    return round(sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) / 1e6, 2)


def latency(model, stories: pd.DataFrame) -> dict:
    """Seconds to predict a 50-story backlog, encoding the text from scratch (the model is already loaded)."""
    sample = stories.sample(LATENCY_STORIES, random_state=SEED)
    times = []
    for run in range(LATENCY_RUNS + 2):
        started = time.perf_counter()
        if isinstance(model, predictor.StackPredictor):
            # each encoder once, even when several bases share it (as the service would do)
            by_encoder = {base.encoder_name: base for base in model.bases.values()}
            text = {name: base.encode(sample, fresh=True) for name, base in by_encoder.items()}
        else:
            text = model.encode(sample, fresh=True)
        model.predict(sample, text)
        if run >= 2:  # two warm-up runs
            times.append(time.perf_counter() - started)
    return {"stories": LATENCY_STORIES, "runs": LATENCY_RUNS, "p50_seconds": round(float(np.median(times)), 4),
            "p95_seconds": round(float(np.percentile(times, 95)), 4), "hardware": "development laptop CPU"}


def base_manifest(config: Config, arena: ArenaData) -> dict:
    encoder_spec = (SbertEncoder().save(Path()) if config.encoder == "sbert" else json.loads(
        (data.ENCODERS_DIR / config.encoder / "encoder.json").read_text(encoding="utf-8")) | {
        "dir": f"encoders/{config.encoder}"})
    return {
        "config": asdict(config), "arena": ARENA, "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(), "encoder": encoder_spec, "levels": arena.levels,
        "data": {"source": "TAWOS v1.1", "stories": len(arena.frame), "split": arena.parts.value_counts().to_dict(),
                 "split_rule": "per project by sprint start: oldest 60% train, next 20% calibration, newest 20% test",
                 "risk_labels": "automatic R1-R6, before the 200-story human check"},
    }


def save_predictions(config: Config, arena: ArenaData, log_points, raw, fold_outputs) -> None:
    """All stories' predictions, and the inner-fold (out-of-fold) predictions of the best trials."""
    frame = pd.DataFrame({"split": arena.parts, "effort_log": log_points, "risk_raw": raw}, index=arena.frame.index)
    (WORK_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    frame.to_parquet(WORK_DIR / "predictions" / f"{config.id}.parquet")
    train = arena.train
    parts = []
    for i, (_, val) in enumerate(arena.folds):
        parts.append(pd.DataFrame({"fold": i + 1, "effort_log": fold_outputs["effort"][i],
                                   "risk_raw": fold_outputs["risk"][i]}, index=train.index[val]))
    (WORK_DIR / "oof").mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_parquet(WORK_DIR / "oof" / f"{config.id}.parquet")


def save_trials(config: Config, name: str, study) -> None:
    (WORK_DIR / "trials").mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(WORK_DIR / "trials" / f"{config.id}-{name}.csv", index=False)


def verify(directory: Path, arena: ArenaData, full_text: np.ndarray, log_points, raw) -> dict:
    """Reload from disk: the saved configuration must reproduce every prediction exactly."""
    loaded = predictor.load(directory)
    again_log, again_raw = loaded.raw(arena.frame, full_text)
    if not (np.array_equal(again_log, log_points) and np.array_equal(again_raw, raw)):
        raise AssertionError(f"{directory.name}: the reloaded model does not reproduce its predictions")
    return {"round_trip": "identical predictions for all stories after reloading from disk",
            "size_mb": directory_size(directory)}


def run_single(config: Config, arena: ArenaData, trials: int) -> None:
    learner = LEARNERS[config.learner]
    full_text = arena.full_text(config.encoder)
    columns = text_columns(config.encoder)
    tr, cal = arena.mask("train"), arena.mask("cal")
    manifest, outputs, fold_outputs, models = base_manifest(config, arena), {}, {}, {}
    for task in ("effort", "risk"):
        log(f"{config.id}: tuning {task}")
        study, fold_predictions, tuning = tune_single(config, task, arena, trials)
        save_trials(config, task, study)
        x = design(arena.frame, full_text, task, text_columns=columns, levels=arena.levels)
        y = targets(arena.frame, task)
        started = time.perf_counter()
        models[task] = learner(task).fit(x[tr], y[tr], x[cal], y[cal], study.best_params)
        manifest[task] = {"inputs": list(x.columns), "tuning": tuning,
                          "fit_seconds": round(time.perf_counter() - started, 1)}
        outputs[task] = models[task].predict(x)
        fold_outputs[task] = fold_predictions
        log(f"{config.id}: {task} trained (inner-fold {tuning['metric']} {tuning['best_value']:.4f})")
    write(config, arena, manifest, models, outputs["effort"], outputs["risk"], fold_outputs, full_text)


def run_joint(config: Config, arena: ArenaData, trials: int) -> None:
    full_text = arena.full_text(config.encoder)
    tr, cal = arena.mask("train"), arena.mask("cal")
    log(f"{config.id}: tuning")
    study, fold_predictions, tuning = tune_joint(config, arena, trials)
    save_trials(config, "joint", study)
    x = design(arena.frame, full_text, "joint", text_columns=text_columns(config.encoder), levels=arena.levels)
    y = {task: targets(arena.frame, task) for task in ("effort", "risk")}
    started = time.perf_counter()
    model = MLP(("effort", "risk")).fit(x[tr], {t: v[tr] for t, v in y.items()}, x[cal],
                                        {t: v[cal] for t, v in y.items()}, study.best_params)
    manifest = base_manifest(config, arena)
    manifest["tuning"] = tuning
    manifest["fit_seconds"] = round(time.perf_counter() - started, 1)
    manifest["effort"], manifest["risk"] = {"inputs": list(x.columns)}, {"inputs": list(x.columns)}
    out = model.predict(x)
    folds = {task: [p[task] for p in fold_predictions] for task in ("effort", "risk")}
    write(config, arena, manifest, {"joint": model}, out["effort"], out["risk"], folds, full_text)


def write(config, arena, manifest, models, log_points, raw, fold_outputs, full_text) -> None:
    fitted = uncertainty(arena, log_points, raw)
    directory = MODELS_DIR / config.id
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for name, model in models.items():
        spec = model.save(directory, name)
        if name == "joint":
            manifest["joint"] = spec
        else:
            manifest[name]["model"] = spec
    manifest["effort"]["intervals"] = fitted["intervals"].to_dict()
    manifest["risk"].update({"calibrator": fitted["calibrator"].to_dict(), "threshold": fitted["threshold"],
                             "bands": dict(RISK_BANDS)})
    path = directory / predictor.MANIFEST
    path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    save_predictions(config, arena, log_points, raw, fold_outputs)
    manifest["check"] = verify(directory, arena, full_text, log_points, raw)
    path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    log(f"{config.id}: saved to {directory} ({manifest['check']['size_mb']} MB), reload check passed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", choices=configs.TRAINED_BY_ARENA, default=configs.TRAINED_BY_ARENA)
    parser.add_argument("--trials", type=int, default=configs.N_TRIALS)
    parser.add_argument("--prepare", action="store_true", help="only fit and cache the text encoders")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    arena = ArenaData.load()
    log(f"{len(arena.frame):,} stories; inner folds validate on "
        f"{', '.join(str(int(v.sum())) for _, v in arena.folds)} training stories")
    for name in dict.fromkeys(configs.BY_ID[c].encoder for c in args.configs):
        for fold in range(len(arena.folds)):
            arena.fold_text(name, fold)
        arena.full_text(name)
        log(f"encoder {name} ready")
    if args.prepare:
        return
    for config_id in args.configs:
        config = configs.BY_ID[config_id]
        if (MODELS_DIR / config.id / predictor.MANIFEST).exists() and not args.force:
            log(f"{config.id}: already saved, skipped (use --force to retrain)")
            continue
        (run_joint if config.joint else run_single)(config, arena, args.trials)


if __name__ == "__main__":
    main()
