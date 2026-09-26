"""Phase 4: the arena's leaderboard, its statistics and its report (ML guide section 8, step 9).

Every configuration is scored once on the test split (the newest 20% of each project's sprints), with the
calibration and intervals it fitted on the calibration split. Then:
  - significance: effort errors against the best configuration with the Wilcoxon signed-rank test, risk
    ROC-AUC differences with a paired bootstrap, Holm-corrected for the many comparisons, plus the
    Vargha-Delaney A12 effect size (ML guide 9);
  - the NFR checks: NFR1 latency, NFR2 accuracy, NFR3 calibration and interval coverage;
  - the composite score the router (R1) will use, with the ML guide's starting weights, a check that the winner
    holds under other reasonable weights, and the winner per project.
The 50-story latency is timed here, for all configurations one after the other on an otherwise idle machine.

Usage:
    erp-arena-report    # after erp-train-arena; builds the stack if it is missing, then writes
                        #   ml-engine/models/arena-v1/stack/             the stacked ensemble
                        #   ml-engine/models/arena-v1/leaderboard.json   what the router (Phase 5) reads
                        #   ml-engine/reports/arena.md                   leaderboard, statistics, NFR checks
                        # and logs every configuration to MLflow in Datasets/effort-risk/mlflow/
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from erp import config as paths
from erp.arena import configs, predictor, stack
from erp.arena.data import ENCODERS_DIR, MODELS_DIR, WORK_DIR, ArenaData
from erp.arena.train import directory_size, latency, log
from erp.explore.profile_tawos import md_table
from erp.models import metrics
from erp.models.bundle import INTERVAL_COVERAGES, code_version
from erp.models.calibration import Calibrator, coverage_and_width, load_intervals

WEIGHTS = {"effort_sa": 0.35, "risk_f1": 0.25, "calibration": 0.25, "latency": 0.15}  # ML guide 4.8
ALTERNATIVE_WEIGHTS = {
    "equal": {"effort_sa": 0.25, "risk_f1": 0.25, "calibration": 0.25, "latency": 0.25},
    "effort first": {"effort_sa": 0.5, "risk_f1": 0.2, "calibration": 0.2, "latency": 0.1},
    "risk first": {"effort_sa": 0.2, "risk_f1": 0.4, "calibration": 0.3, "latency": 0.1},
    "no latency": {"effort_sa": 0.4, "risk_f1": 0.3, "calibration": 0.3, "latency": 0.0},
}
LATENCY_BUDGET = 2.0  # NFR1: a 50-story backlog within 2 s at the 95th percentile
ECE_LIMIT = 0.10  # NFR3
COVERAGE_TOLERANCE = 0.05  # NFR3: coverage within 5 percentage points of the nominal level
F1_TARGET = 0.75  # NFR2
IMPROVEMENT_TARGET = 0.20  # NFR2: at least 20% better than the mean and random baselines
MIN_PROJECT_TEST = 30  # projects with fewer test stories get the pooled winner
MLFLOW_DIR = paths.WORK_DIR / "mlflow"
REPORT = paths.REPO_ROOT / "ml-engine" / "reports" / "arena.md"


def saved_configs() -> dict[str, dict]:
    found = {}
    for config in configs.CONFIGS:
        path = MODELS_DIR / config.id / predictor.MANIFEST
        if path.exists() and (WORK_DIR / "predictions" / f"{config.id}.parquet").exists():
            found[config.id] = json.loads(path.read_text(encoding="utf-8"))
    return found


def outputs(arena: ArenaData, name: str, manifest: dict) -> pd.DataFrame:
    """Per story: log points, points, raw score and calibrated probability."""
    stored = pd.read_parquet(WORK_DIR / "predictions" / f"{name}.parquet").loc[arena.frame.index]
    calibrator = Calibrator.from_dict(manifest["risk"]["calibrator"])
    return stored.assign(points=np.expm1(stored["effort_log"]), probability=calibrator.predict(stored["risk_raw"]))


def effort_metrics(stories: pd.DataFrame, out: pd.DataFrame, manifest: dict, baselines: dict,
                   directory: Path) -> dict:
    result = metrics.effort_report(stories["story_points"], out["points"], stories["Project_ID"])
    intervals = load_intervals(manifest["effort"]["intervals"], directory, manifest.get("levels"))
    for coverage in INTERVAL_COVERAGES:
        low, high = intervals.bounds(stories, out["effort_log"].to_numpy(), coverage)
        result[f"coverage_{coverage:g}"], result[f"width_{coverage:g}"] = coverage_and_width(
            stories["story_points"], low, high)
    for name, mae in baselines.items():
        result[f"improvement_vs_{name}"] = 1 - result["mae"] / mae
    return result


def risk_metrics(stories: pd.DataFrame, out: pd.DataFrame, threshold: float) -> dict:
    result = metrics.classification_report(stories["at_risk"], out["probability"], threshold)
    result["ece_before"] = metrics.expected_calibration_error(stories["at_risk"], out["risk_raw"])
    return result


def baselines(arena: ArenaData) -> tuple[dict, dict, dict]:
    """Effort: mean / median of the project's training stories and random guessing; risk: majority class and
    the team's recent spillover rate (threshold chosen on the calibration split), as in Phase 2."""
    frame, test = arena.frame, arena.frame[arena.mask("test")]
    train_points = frame[arena.mask("train")].groupby("Project_ID")["story_points"]
    effort_guess = {"mean": test["Project_ID"].map(train_points.mean()),
                    "median": test["Project_ID"].map(train_points.median())}
    effort = {name: metrics.effort_report(test["story_points"], guess, test["Project_ID"])
              for name, guess in effort_guess.items()}
    effort["random"] = {"mae": metrics.random_guessing_mae(test["story_points"], test["Project_ID"]),
                        "mdae": np.nan, "sa": 0.0}
    prior = float(frame.loc[arena.mask("train"), "at_risk"].mean())
    rate = frame["historical_spillover_rate"].fillna(prior)
    cal = arena.mask("cal")
    threshold = metrics.choose_threshold(frame["at_risk"][cal], rate[cal])
    risk = {"spillover rate": metrics.classification_report(test["at_risk"], rate[arena.mask("test")], threshold)}
    return effort, risk, {"median": effort_guess["median"], "spillover rate": rate[arena.mask("test")]}


def size_on_disk(name: str, manifest: dict, found: dict) -> float:
    if name == "stack":
        bases = set(manifest["effort"]["bases"]) | set(manifest["risk"]["bases"])
        return round(directory_size(MODELS_DIR / name) + sum(size_on_disk(b, found[b], found) for b in bases), 2)
    encoder = manifest["encoder"]["name"]
    shared = directory_size(ENCODERS_DIR / encoder) if (ENCODERS_DIR / encoder).exists() else 0.0
    return round(directory_size(MODELS_DIR / name) + shared, 2)


def tuning_cost(name: str, manifest: dict, found: dict) -> tuple[float, float]:
    """(tuning minutes, final fit seconds); the stack's include its bases'."""
    if name == "stack":
        bases = set(manifest["effort"]["bases"]) | set(manifest["risk"]["bases"])
        costs = [tuning_cost(b, found[b], found) for b in bases]
        return round(sum(c[0] for c in costs), 1), round(sum(c[1] for c in costs), 1)
    if "tuning" in manifest:  # M3
        return round(manifest["tuning"]["seconds"] / 60, 1), manifest["fit_seconds"]
    return (round(sum(manifest[t]["tuning"]["seconds"] for t in ("effort", "risk")) / 60, 1),
            round(sum(manifest[t]["fit_seconds"] for t in ("effort", "risk")), 1))


def composite(row: dict, weights: dict) -> float:
    latency_score = max(0.0, 1 - row["latency_p95"] / LATENCY_BUDGET)
    return (weights["effort_sa"] * row["sa"] / 100 + weights["risk_f1"] * row["f1"]
            + weights["calibration"] * (1 - row["ece"]) + weights["latency"] * latency_score)


def failed_requirements(row: dict) -> list[str]:
    failed = []
    if row["latency_p95"] > LATENCY_BUDGET:
        failed.append("NFR1 latency")
    if row["ece"] >= ECE_LIMIT:
        failed.append("NFR3 ECE")
    for coverage in INTERVAL_COVERAGES:
        if abs(row[f"coverage_{coverage:g}"] - coverage) > COVERAGE_TOLERANCE:
            failed.append(f"NFR3 {coverage:.0%} coverage")
    return failed


def winner(scores: dict[str, float], eligible: set[str]) -> str | None:
    candidates = {n: s for n, s in scores.items() if n in eligible}
    return max(candidates, key=candidates.get) if candidates else None


def per_project(arena: ArenaData, results: dict, table: pd.DataFrame, eligible: set[str]) -> dict:
    test = arena.frame[arena.mask("test")]
    keys = arena.frame.groupby("Project_ID")["project_key"].first()
    projects = {}
    for project, stories in test.groupby("Project_ID"):
        if len(stories) < MIN_PROJECT_TEST:
            continue
        scores, detail = {}, {}
        for name, result in results.items():
            out = result["out"].loc[stories.index]
            effort = metrics.effort_report(stories["story_points"], out["points"], stories["Project_ID"])
            risk = metrics.classification_report(stories["at_risk"], out["probability"], result["threshold"])
            row = {"sa": effort["sa"], "f1": risk["f1"], "ece": risk["ece"],
                   "latency_p95": table.loc[name, "latency_p95"]}
            scores[name] = round(composite(row, WEIGHTS), 4)
            detail[name] = {"mae": round(effort["mae"], 3), "sa": round(effort["sa"], 1),
                            "f1": round(risk["f1"], 3), "roc_auc": round(risk["roc_auc"], 3),
                            "ece": round(risk["ece"], 3)}
        best = winner(scores, eligible)
        projects[str(keys[project])] = {"test_stories": len(stories), "at_risk": round(float(stories["at_risk"].mean()),
                                                                                     3),
                                        "winner": best, "composite": scores, "metrics": detail}
    return projects


def significance(arena: ArenaData, results: dict, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import roc_auc_score

    test = arena.frame[arena.mask("test")]
    errors = {n: (test["story_points"] - r["out"].loc[test.index, "points"]).abs() for n, r in results.items()}
    best_effort = table["mae"].idxmin()
    rows = []
    for name in table.index.drop(best_effort):
        rows.append({"config": name, "vs": best_effort, "difference": table.loc[name, "mae"] - table.loc[
            best_effort, "mae"], "p": metrics.wilcoxon_p(errors[name], errors[best_effort]),
            "a12": metrics.vargha_delaney(errors[name], errors[best_effort])})
    effort = pd.DataFrame(rows)
    if len(effort):
        effort["p_holm"] = metrics.holm(effort["p"])
    best_risk = table["roc_auc"].idxmax()
    rows = []
    for name in table.index.drop(best_risk):
        boot = metrics.paired_bootstrap(roc_auc_score, test["at_risk"].to_numpy(),
                                        results[name]["out"].loc[test.index, "probability"],
                                        results[best_risk]["out"].loc[test.index, "probability"])
        rows.append({"config": name, "vs": best_risk, **boot})
    risk = pd.DataFrame(rows)
    if len(risk):
        risk["p_holm"] = metrics.holm(risk["p"])
    return effort, risk


def planned_only(arena: ArenaData, results: dict) -> pd.DataFrame:
    """Sensitivity check agreed on 2026-09-26: the same test metrics on stories planned at sprint start only."""
    test = arena.frame[arena.mask("test")]
    rows = []
    for subset, stories in (("all test stories", test), ("planned only", test[~test["added_mid_sprint"]]),
                            ("added mid-sprint", test[test["added_mid_sprint"]])):
        for name, result in results.items():
            out = result["out"].loc[stories.index]
            effort = metrics.effort_report(stories["story_points"], out["points"], stories["Project_ID"])
            risk = metrics.classification_report(stories["at_risk"], out["probability"], result["threshold"])
            rows.append({"subset": subset, "config": name, "stories": len(stories), "mae": effort["mae"],
                         "sa": effort["sa"], "roc_auc": risk["roc_auc"], "f1": risk["f1"], "ece": risk["ece"]})
    return pd.DataFrame(rows)


def log_mlflow(found: dict, table: pd.DataFrame, created: str) -> str:
    """One MLflow run per configuration: its best settings, tuning, test metrics and costs."""
    import mlflow

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}")
    name = f"effort-risk-{configs.ARENA}"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(MLFLOW_DIR / "artifacts").as_uri())
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"arena report {created}"):
        mlflow.set_tags({"code_commit": code_version()["commit"], "arena": configs.ARENA})
        for config_id, manifest in found.items():
            with mlflow.start_run(run_name=config_id, nested=True):
                cfg = configs.BY_ID[config_id]
                mlflow.set_tags({"encoder": cfg.encoder, "learner": cfg.learner, "role": cfg.role,
                                 "code_commit": manifest.get("code", {}).get("commit", "")})
                params = {}
                for part in ("effort", "risk"):
                    params.update({f"{part}.{k}": v for k, v in manifest[part].get("tuning", {}).get(
                        "best_params", {}).items()})
                params.update({f"joint.{k}": v for k, v in manifest.get("tuning", {}).get("best_params", {}).items()})
                if params:
                    mlflow.log_params(params)
                row = table.loc[config_id]
                mlflow.log_metrics({k: float(row[k]) for k in (
                    "mae", "mdae", "sa", "coverage_0.8", "coverage_0.9", "roc_auc", "pr_auc", "f1", "precision",
                    "recall", "brier", "ece", "latency_p95", "size_mb", "tuning_minutes", "composite")
                    if pd.notna(row[k])})
                for trials in sorted((WORK_DIR / "trials").glob(f"{config_id}-*.csv")):
                    mlflow.log_artifact(str(trials), "optuna-trials")
    return name


def fmt(value, digits: int = 3) -> str:
    return "–" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:,.{digits}f}"


def build_report(arena, table, effort_base, risk_base, sig_effort, sig_risk, sensitivity, projects,
                 planned, found, phase3, eligible) -> str:
    labels = {c.id: c.label for c in configs.CONFIGS}
    missing = [c.label + (" (fine-tuning needs the GPU; it joins the leaderboard when trained)"
                          if c.id == "distilbert" else "") for c in configs.CONFIGS if c.id not in found]
    ordered = table.sort_values("mae")
    effort_rows = [{"Configuration": labels[n], "MAE": fmt(r["mae"], 2), "MdAE": fmt(r["mdae"], 2),
                    "SA": fmt(r["sa"], 1), "vs mean guess": f"{r['improvement_vs_mean']:+.0%}",
                    "80% interval: coverage": f"{r['coverage_0.8']:.1%}", "width": fmt(r["width_0.8"], 1),
                    "90% interval: coverage": f"{r['coverage_0.9']:.1%}", "width ": fmt(r["width_0.9"], 1)}
                   for n, r in ordered.iterrows()]
    blank = {"80% interval: coverage": "–", "width": "–", "90% interval: coverage": "–", "width ": "–"}
    for name, label in (("median", "Median of the project's training stories"),
                        ("mean", "Mean of the project's training stories"), ("random", "Random guessing")):
        b = effort_base[name]
        effort_rows.append({"Configuration": f"*{label}*", "MAE": fmt(b["mae"], 2), "MdAE": fmt(b["mdae"], 2),
                            "SA": fmt(b["sa"], 1), "vs mean guess": f"{1 - b['mae'] / effort_base['mean']['mae']:+.0%}",
                            **blank})
    m1 = phase3["m1"]
    effort_rows.append({"Configuration": "*Phase 3 bundle (untuned SBERT + LightGBM)*", "MAE": fmt(m1["mae"], 2),
                        "MdAE": fmt(m1["mdae"], 2), "SA": fmt(m1["sa"], 1),
                        "vs mean guess": f"{1 - m1['mae'] / effort_base['mean']['mae']:+.0%}",
                        "80% interval: coverage": f"{m1['coverage_0.8']:.1%}", "width": fmt(m1["mean_width_0.8"], 1),
                        "90% interval: coverage": f"{m1['coverage_0.9']:.1%}", "width ": fmt(m1["mean_width_0.9"], 1)})

    risk_order = table.sort_values("roc_auc", ascending=False)

    def risk_row(label, r):
        return {"Configuration": label, "ROC-AUC": fmt(r["roc_auc"]), "PR-AUC": fmt(r["pr_auc"]), "F1": fmt(r["f1"]),
                "Precision": fmt(r["precision"]), "Recall": fmt(r["recall"]), "Brier": fmt(r["brier"]),
                "ECE before C1": fmt(r.get("ece_before", np.nan)), "ECE after C1": fmt(r["ece"]),
                "Threshold": fmt(r["threshold"])}

    risk_rows = [risk_row(labels[n], r) for n, r in risk_order.iterrows()]
    risk_rows.append(risk_row("*Team's recent spillover rate*", {**risk_base["spillover rate"],
                                                                 "ece_before": np.nan}))
    m2 = phase3["m2"]
    risk_rows.append(risk_row("*Phase 3 bundle (untuned SBERT + LightGBM)*", {
        **m2["after_calibration"], "ece_before": m2["before_calibration"]["ece"]}))

    test = arena.frame[arena.mask("test")]
    best_effort, best_risk = ordered.index[0], risk_order.index[0]

    config_rows = []
    for c in configs.CONFIGS:
        if c.id not in found:
            continue
        m = found[c.id]
        if c.id == "stack":
            tuning = f"none of its own; bases chosen by inner-fold score (effort: {', '.join(m['effort']['bases'])};" \
                     f" risk: {', '.join(m['risk']['bases'])})"
        elif "tuning" in m:
            t = m["tuning"]
            tuning = f"{t['complete']} complete + {t['pruned']} pruned; validation loss {t['best_value']:.4f}"
        else:
            e, r = m["effort"]["tuning"], m["risk"]["tuning"]
            tuning = (f"{e['complete'] + r['complete']} complete + {e['pruned'] + r['pruned']} pruned; inner-fold "
                      f"MAE {e['best_value']:.3f}, ROC-AUC {r['best_value']:.3f}")
        config_rows.append({"Configuration": c.label, "Role (Appendix D)": c.role, "Tuning": tuning,
                            "Tuning time": f"{table.loc[c.id, 'tuning_minutes']:.0f} min"})

    sig_effort_rows = [{"Configuration": labels[r["config"]], "MAE difference": f"{r['difference']:+.3f}",
                        "Wilcoxon p (Holm)": f"{r['p_holm']:.2g}", "A12": f"{r['a12']:.3f}",
                        "Effect": metrics.effect_size_label(r["a12"])} for _, r in sig_effort.iterrows()]
    sig_risk_rows = [{"Configuration": labels[r["config"]], "ROC-AUC difference": f"{r['difference']:+.3f}",
                      "95% interval": f"{r['low']:+.3f} to {r['high']:+.3f}",
                      "Bootstrap p (Holm)": f"{r['p_holm']:.2g}"}
                     for _, r in sig_risk.iterrows()]

    def check(ok: bool) -> str:
        return "yes" if ok else "**no**"

    nfr_rows = []
    for n, r in table.iterrows():
        nfr_rows.append({
            "Configuration": labels[n],
            "NFR1: 50 stories, p95": f"{r['latency_p95']:.2f} s ({check(r['latency_p95'] <= LATENCY_BUDGET)})",
            "NFR2: SA ≥ 20": f"{r['sa']:.1f} ({check(r['sa'] >= 100 * IMPROVEMENT_TARGET)})",
            "NFR2: ≥ 20% better than mean guess": f"{r['improvement_vs_mean']:+.0%} "
                                                  f"({check(r['improvement_vs_mean'] >= IMPROVEMENT_TARGET)})",
            "NFR2: F1 ≥ 0.75": f"{r['f1']:.3f} ({check(r['f1'] >= F1_TARGET)})",
            "NFR3: ECE < 0.10": f"{r['ece']:.3f} ({check(r['ece'] < ECE_LIMIT)})",
            "NFR3: coverage ±5 pp": check(all(abs(r[f'coverage_{c:g}'] - c) <= COVERAGE_TOLERANCE
                                              for c in INTERVAL_COVERAGES)),
        })

    cost_rows = [{"Configuration": labels[n], "Tuning": f"{r['tuning_minutes']:.0f} min",
                  "Final fit": f"{r['fit_seconds']:.0f} s", "On disk": f"{r['size_mb']:.1f} MB",
                  "50 stories p50": f"{r['latency_p50']:.2f} s", "p95": f"{r['latency_p95']:.2f} s"}
                 for n, r in table.sort_values("latency_p95").iterrows()]

    comp = table.sort_values("composite", ascending=False)
    comp_rows = [{"Rank": i + 1, "Configuration": labels[n], "Score": f"{r['composite']:.4f}",
                  "Eligible": "yes" if n in eligible else f"no ({r['failed']})"} for i, (n, r) in
                 enumerate(comp.iterrows())]
    sens_rows = [{"Weights": name, "effort SA / risk F1 / 1 − ECE / latency": " / ".join(
        f"{w[k]:.2f}" for k in ("effort_sa", "risk_f1", "calibration", "latency")),
        "Winner": labels.get(result["winner"], "–")} for name, (w, result) in sensitivity.items()]

    project_rows = []
    for key, p in sorted(projects.items(), key=lambda kv: -kv[1]["test_stories"]):
        w = p["winner"]
        project_rows.append({"Project": key, "Test stories": p["test_stories"], "At risk": f"{p['at_risk']:.0%}",
                             "Winner": labels.get(w, "–"),
                             "MAE": fmt(p["metrics"][w]["mae"], 2) if w else "–",
                             "F1": fmt(p["metrics"][w]["f1"]) if w else "–",
                             "ROC-AUC": fmt(p["metrics"][w]["roc_auc"]) if w else "–"})
    wins = pd.Series([p["winner"] for p in projects.values()]).value_counts()

    top = [best_effort, best_risk] + [n for n in comp.index[:2] if n not in (best_effort, best_risk)]
    planned_rows = [{"Subset": r["subset"], "Configuration": labels[r["config"]], "Stories": r["stories"],
                     "MAE": fmt(r["mae"], 2), "SA": fmt(r["sa"], 1), "ROC-AUC": fmt(r["roc_auc"]),
                     "F1": fmt(r["f1"]), "ECE": fmt(r["ece"])}
                    for _, r in planned[planned["config"].isin(top)].iterrows()]
    pooled = sensitivity["ML guide 4.8"][1]["winner"]

    return "\n".join([
        "# Comparative Model Arena (Phase 4)", "",
        "Generated by `erp-arena-report` after `erp-train-arena`. Every configuration of the proposal's "
        "Appendix D, fixed before any result was seen, trained on the same stories with the same tuning budget, "
        "folds and seeds, and scored once on the same test stories.", "",
        "## 1. How the contest was run", "",
        f"- **Data:** {len(arena.frame):,} stories from TAWOS; per project the oldest 60% of sprints train "
        f"({int(arena.mask('train').sum()):,}), the next 20% calibrate ({int(arena.mask('cal').sum()):,}), the "
        f"newest 20% test ({len(test):,}, {test['at_risk'].mean():.1%} at risk).",
        f"- **Tuning:** Optuna, {configs.N_TRIALS} trials per configuration and task (TPE sampler, seed "
        f"{configs.SEED}; a median pruner stops trials that are clearly worse after the first folds), on "
        f"{configs.INNER_FOLDS} forward-chaining folds inside the training split: each project's training sprints "
        "in 4 blocks, fold i trains on blocks 1..i and validates on block i+1. This is time-aware nested "
        "cross-validation (ML guide 8.1): the calibration and test splits are never seen during tuning. Effort "
        "is tuned for MAE in story points, risk for ROC-AUC, M3 for its validation loss.",
        "- **Encoders fitted on our text** (E1 TF-IDF, E2 FastText) are refitted inside every fold, then on the "
        "whole training split; SBERT (E3) is used as downloaded.",
        "- **Final models** are trained on the whole training split with the best settings; boosters and networks "
        "stop early on the calibration split. On the calibration split each configuration then fits its own C2 "
        "intervals, C1 calibrator and risk threshold (best F1 with recall ≥ 0.70).",
        "- **Every saved configuration was reloaded from disk and reproduced all its predictions exactly**"
        + (" (DistilBERT within 1e-4: its CPU matrix products may round differently between runs)."
           if "distilbert" in found else "."),
        f"- **Not in this run:** {'; '.join(missing) if missing else 'nothing'}.", "",
        md_table(pd.DataFrame(config_rows)), "",
        "## 2. Effort (M1): story points", "",
        "Test split, story points on each project's own scale. SA is against random guessing; 'vs mean guess' is "
        "how much lower the MAE is than always guessing the project's mean (NFR2). Intervals are C2 (adaptive: "
        "normalized split conformal, see `confidence.md`).", "",
        md_table(pd.DataFrame(effort_rows)), "",
        "## 3. Risk (M2): will the story run into trouble?", "",
        "Test split. Probabilities after each configuration's own C1 calibrator; ECE before C1 shows why it is "
        "needed (for the SVM the raw score is a squashed margin, not a probability at all).", "",
        md_table(pd.DataFrame(risk_rows)), "",
        "## 4. Are the differences real?", "",
        f"Effort: every configuration against the most accurate one ({labels[best_effort]}), Wilcoxon signed-rank "
        "test on the paired absolute errors of the same test stories, Holm-corrected for the number of "
        "comparisons. A12 is the Vargha-Delaney effect size: above 0.5 means the configuration's errors tend to be "
        "larger.", "",
        md_table(pd.DataFrame(sig_effort_rows)) if sig_effort_rows else "_Only one configuration._", "",
        f"Risk: ROC-AUC of every configuration minus the best ({labels[best_risk]}), 95% paired bootstrap "
        "interval over test stories (1,000 resamples), Holm-corrected p-values.", "",
        md_table(pd.DataFrame(sig_risk_rows)) if sig_risk_rows else "_Only one configuration._", "",
        "## 5. Non-functional requirements", "",
        "NFR1 is timed here on the development laptop's CPU for one user (encoding the text from scratch, models "
        "already loaded); the 10-concurrent-user load test belongs to the service (Phase 5). NFR2 is read as "
        "'SA ≥ 20 against random guessing and an MAE at least 20% below the mean guess'.", "",
        md_table(pd.DataFrame(nfr_rows)), "",
        "## 6. Cost", "",
        md_table(pd.DataFrame(cost_rows)), "",
        "On disk includes the encoder the configuration shares with others (TF-IDF 12 MB, FastText 24 MB); the "
        "stack includes its bases.", "",
        "## 7. Composite score and what the router will pick", "",
        "The ML guide's starting formula for the router (R1, Phase 5): 0.35 × SA/100 + 0.25 × F1 + 0.25 × "
        f"(1 − ECE) + 0.15 × latency score (1 − p95 / {LATENCY_BUDGET:g} s). A configuration that misses a hard "
        "requirement (NFR1 latency, NFR3 ECE or interval coverage) is not eligible.", "",
        md_table(pd.DataFrame(comp_rows)), "",
        "Does the winner depend on the weights?", "",
        md_table(pd.DataFrame(sens_rows)), "",
        f"Per project (projects with at least {MIN_PROJECT_TEST} test stories; smaller ones use the pooled winner, "
        f"{labels.get(pooled, '–')}). Same formula on the project's own test stories:", "",
        md_table(pd.DataFrame(project_rows)), "",
        "Projects won: " + ", ".join(f"{labels.get(n, n)} {c}" for n, c in wins.items()) + ". Per-project "
        "scores come from few stories and are noisy; the router should prefer the pooled winner unless a project's "
        "winner is clearly ahead.", "",
        "## 8. Sensitivity: planned stories only", "",
        "Stories added after the sprint started are kept (decision of 2026-09-26) with the added_mid_sprint "
        "feature. The same models on the planned stories alone:", "",
        md_table(pd.DataFrame(planned_rows)), "",
        "## 9. Before reading too much into this", "",
        "- The risk labels are automatic (R1–R6) and still wait for the 200-story human check; the arena is rerun "
        "after it with one command.",
        "- The quality, acceptance-criteria and traceability features are proxies for Components 1–3 (H2 is tested "
        "in `hypotheses.md`).",
        "- One pooled model per configuration; per-project training (for projects above 100 stories) is left for "
        "when the router shows it is needed.",
        "- One temporal test split; its stories are the newest sprints of every project, so the numbers include "
        "some drift over time.", "",
    ])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--rebuild-stack", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    arena = ArenaData.load()
    stack.build(arena, force=args.rebuild_stack)
    found = saved_configs()
    test_mask = arena.mask("test")
    test = arena.frame[test_mask]
    effort_base, risk_base, _ = baselines(arena)
    phase3 = json.loads((paths.MODEL_BUNDLES_DIR / "sbert-lightgbm-v1" / "bundle.json").read_text(
        encoding="utf-8"))["test_metrics"]

    results, rows = {}, {}
    for name, manifest in found.items():
        out = outputs(arena, name, manifest)
        threshold = manifest["risk"]["threshold"]
        results[name] = {"out": out, "threshold": threshold}
        effort = effort_metrics(test, out[test_mask], manifest, {"mean": effort_base["mean"]["mae"],
                                                                 "median": effort_base["median"]["mae"]},
                                MODELS_DIR / name)
        risk = risk_metrics(test, out[test_mask], threshold)
        log(f"timing {name}")
        timing = latency(predictor.load(MODELS_DIR / name), test)
        tuning_minutes, fit_seconds = tuning_cost(name, manifest, found)
        rows[name] = {**effort, **risk, "latency_p50": timing["p50_seconds"], "latency_p95": timing["p95_seconds"],
                      "size_mb": size_on_disk(name, manifest, found), "tuning_minutes": tuning_minutes,
                      "fit_seconds": fit_seconds}
    table = pd.DataFrame(rows).T.astype(float)
    table["failed"] = [", ".join(failed_requirements(r)) for _, r in table.iterrows()]
    eligible = {n for n, r in table.iterrows() if not r["failed"]}
    table["composite"] = [composite(r, WEIGHTS) for _, r in table.iterrows()]
    sensitivity = {"ML guide 4.8": (WEIGHTS, None), **{k: (w, None) for k, w in ALTERNATIVE_WEIGHTS.items()}}
    sensitivity = {name: (w, {"winner": winner({n: composite(r, w) for n, r in table.iterrows()}, eligible)})
                   for name, (w, _) in sensitivity.items()}
    projects = per_project(arena, results, table, eligible)
    sig_effort, sig_risk = significance(arena, results, table)
    planned = planned_only(arena, results)

    created = datetime.now(UTC).isoformat(timespec="seconds")
    labels = {c.id: c for c in configs.CONFIGS}
    leaderboard = {
        "arena": configs.ARENA, "created": created, "code": code_version(),
        "test_split": {"stories": len(test), "at_risk": round(float(test["at_risk"].mean()), 4)},
        "weights": WEIGHTS, "hard_requirements": {"latency_p95_seconds": LATENCY_BUDGET, "ece_below": ECE_LIMIT,
                                                  "coverage_tolerance": COVERAGE_TOLERANCE},
        "pooled_winner": sensitivity["ML guide 4.8"][1]["winner"],
        "sensitivity": {name: result["winner"] for name, (_, result) in sensitivity.items()},
        "min_project_test_stories": MIN_PROJECT_TEST,
        "per_project": projects,
        "configs": {n: {"label": labels[n].label, "role": labels[n].role, "encoder": labels[n].encoder,
                        "learner": labels[n].learner, "eligible": n in eligible,
                        "failed_requirements": r["failed"].split(", ") if r["failed"] else [],
                        **{k: (round(float(v), 4) if isinstance(v, int | float | np.floating) else v)
                           for k, v in r.items() if k != "failed"}}
                    for n, r in table.iterrows()},
    }
    (MODELS_DIR / "leaderboard.json").write_text(json.dumps(leaderboard, indent=1, default=str), encoding="utf-8")
    args.report.write_text(build_report(arena, table, effort_base, risk_base, sig_effort, sig_risk, sensitivity,
                                        projects, planned, found, phase3, eligible), encoding="utf-8")
    if not args.no_mlflow:
        log(f"MLflow experiment {log_mlflow(found, table, created)} in {MLFLOW_DIR}")
    log(f"Wrote {args.report} and {MODELS_DIR / 'leaderboard.json'}")


if __name__ == "__main__":
    main()
