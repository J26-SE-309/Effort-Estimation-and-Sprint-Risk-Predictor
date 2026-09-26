"""How reliable are predictions for a project the models never saw, before and after it has sprint history?
Evidence for the confidence score's unseen-project penalty (erp/models/confidence.py).

The models learned from the TAWOS projects, so every platform project is new to them, and every platform
prediction carries the x0.7 unseen-project penalty for good, even once the service holds the project's own sprint
history. This experiment measures what being unseen costs, and whether the project's history takes that cost away.

Leave one project out: for each TAWOS project P with test stories, the router's configuration (FastText +
LightGBM, with its tuned settings) is trained without P's stories, and P's project key is unknown to it, as for a
platform project; it then predicts every story of P. The same configuration trained with P predicts P's test
stories. Groups compared:

  seen, test stories           the models knew the project (the arena's situation)
  unseen, test stories         the same stories, the project unknown: the cost of being unseen
  unseen, cold / warm          all of P's stories, before / from its 3rd closed sprint (the cold-start rule)

    erp-unseen-projects                 -> ml-engine/reports/unseen-projects.md
    erp-unseen-projects --report-only   # from the saved predictions (Datasets/effort-risk/arena/arena-v1/unseen)

The FastText encoder was fitted on every project's training stories, P's included: for unseen projects the text
side is slightly optimistic.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from erp import config as paths
from erp.arena import configs, train
from erp.arena.data import MODELS_DIR, WORK_DIR, ArenaData, targets, text_columns
from erp.explore.profile_tawos import md_table
from erp.models import calibration, metrics
from erp.models.confidence import COLD_START_SPRINTS, PENALTIES
from erp.models.inputs import design
from erp.models.learners import LEARNERS

CONFIG = "fasttext-lightgbm"
UNSEEN = "__unseen__"
OUT_DIR = WORK_DIR / "unseen"
REPORT = paths.REPO_ROOT / "ml-engine" / "reports" / "unseen-projects.md"


def _fit_predict(arena: ArenaData, frame: pd.DataFrame, text: np.ndarray, task: str, fit: np.ndarray,
                 stop: np.ndarray, params: dict) -> np.ndarray:
    """The configuration trained on `fit` (early stopping on `stop`), predicting every story of `frame`."""
    x = design(frame, text, task, text_columns=text_columns(configs.BY_ID[CONFIG].encoder), levels=arena.levels)
    y = targets(frame, task)
    learner = LEARNERS[configs.BY_ID[CONFIG].learner](task).fit(x[fit], y[fit], x[stop], y[stop], params,
                                                                  seed=configs.SEED)
    return learner.predict(x)


def _probability(raw: np.ndarray, y: np.ndarray, calibrate: np.ndarray) -> np.ndarray:
    return calibration.Calibrator("auto").fit(raw[calibrate], y[calibrate]).predict(raw)


def run(arena: ArenaData) -> pd.DataFrame:
    """Per story of every held-out project: points and at-risk truth, and the seen / unseen predictions."""
    manifest = json.loads((MODELS_DIR / CONFIG / "model.json").read_text(encoding="utf-8"))
    params = {task: manifest[task]["tuning"]["best_params"] for task in ("effort", "risk")}
    frame, text = arena.frame, arena.full_text(configs.BY_ID[CONFIG].encoder)
    tr, cal, test = arena.mask("train"), arena.mask("cal"), arena.mask("test")
    y_risk = targets(frame, "risk")
    project = frame["project_key"].astype(str).to_numpy()

    seen_effort = _fit_predict(arena, frame, text, "effort", tr, cal, params["effort"])
    seen_risk = _probability(_fit_predict(arena, frame, text, "risk", tr, cal, params["risk"]), y_risk, cal)
    train.log("seen models done")

    held_out = sorted(set(project[test]))
    unseen_effort, unseen_risk = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    for number, key in enumerate(held_out, 1):
        mine = project == key
        hidden = frame.assign(project_key=np.where(mine, UNSEEN, frame["project_key"].astype(str)))
        unseen_effort[mine] = _fit_predict(arena, hidden, text, "effort", tr & ~mine, cal & ~mine,
                                           params["effort"])[mine]
        raw = _fit_predict(arena, hidden, text, "risk", tr & ~mine, cal & ~mine, params["risk"])
        unseen_risk[mine] = _probability(raw, y_risk, cal & ~mine)[mine]
        train.log(f"{key} held out ({number}/{len(held_out)})")

    return pd.DataFrame({
        "Issue_ID": frame.index.to_numpy(), "project": project, "split": frame["split"].to_numpy(),
        "history_sprints": frame["history_sprints"].to_numpy(dtype=float),
        "points": frame["story_points"].to_numpy(dtype=float), "at_risk": y_risk.astype(bool),
        "seen_points": np.expm1(seen_effort), "seen_risk": seen_risk,
        "unseen_points": np.expm1(unseen_effort), "unseen_risk": unseen_risk,
    })[np.isin(project, held_out)].reset_index(drop=True)


def _scores(points, predicted, at_risk, probability) -> dict:
    at_risk = np.asarray(at_risk, bool)
    return {"stories": len(points), "mae": metrics.mae(points, predicted),
            "brier": float(brier_score_loss(at_risk, probability)),
            "roc_auc": float(roc_auc_score(at_risk, probability)) if 0 < at_risk.mean() < 1 else float("nan"),
            "ece": metrics.expected_calibration_error(at_risk, probability)}


def summarise(stories: pd.DataFrame) -> dict:
    test = stories[stories["split"] == "test"]
    warm = stories["history_sprints"].fillna(0) >= COLD_START_SPRINTS
    groups = {
        "seen, test stories": _scores(test["points"], test["seen_points"], test["at_risk"], test["seen_risk"]),
        "unseen, test stories": _scores(test["points"], test["unseen_points"], test["at_risk"], test["unseen_risk"]),
        f"unseen, cold (< {COLD_START_SPRINTS} sprints)": _scores(
            stories.loc[~warm, "points"], stories.loc[~warm, "unseen_points"], stories.loc[~warm, "at_risk"],
            stories.loc[~warm, "unseen_risk"]),
        f"unseen, warm (>= {COLD_START_SPRINTS} sprints)": _scores(
            stories.loc[warm, "points"], stories.loc[warm, "unseen_points"], stories.loc[warm, "at_risk"],
            stories.loc[warm, "unseen_risk"]),
    }
    y = test["at_risk"].to_numpy(bool)
    cost = {  # unseen minus seen, on the same test stories (positive: unseen is worse for MAE and Brier)
        "mae": metrics.paired_bootstrap(metrics.mae, test["points"], test["unseen_points"], test["seen_points"]),
        "brier": metrics.paired_bootstrap(brier_score_loss, y, test["unseen_risk"], test["seen_risk"]),
        "roc_auc": metrics.paired_bootstrap(roc_auc_score, y, test["unseen_risk"], test["seen_risk"]),
    }
    per_project = []
    for key, own in test.groupby("project"):
        row = {"project": key, "test stories": len(own),
               "MAE seen": metrics.mae(own["points"], own["seen_points"]),
               "MAE unseen": metrics.mae(own["points"], own["unseen_points"]),
               "Brier seen": float(brier_score_loss(own["at_risk"], own["seen_risk"])),
               "Brier unseen": float(brier_score_loss(own["at_risk"], own["unseen_risk"]))}
        per_project.append(row)
    seen = groups["seen, test stories"]
    support = {  # how much of the seen models' accuracy each group keeps: the size a data-support penalty warrants
        name: {"effort (seen MAE / MAE)": seen["mae"] / g["mae"],
               "risk (ROC-AUC above chance / seen's)": (g["roc_auc"] - 0.5) / (seen["roc_auc"] - 0.5)}
        for name, g in groups.items() if name != "seen, test stories"}
    return {"groups": groups, "cost_of_unseen": cost, "support": support, "per_project": per_project,
            "projects": int(stories["project"].nunique()), "config": CONFIG,
            "created": datetime.now(UTC).isoformat(timespec="seconds")}


def build_report(summary: dict) -> str:
    groups = pd.DataFrame([{"Group": name, "Stories": f"{g['stories']:,}", "Effort MAE (points)": f"{g['mae']:.3f}",
                            "Risk Brier": f"{g['brier']:.4f}", "Risk ROC-AUC": f"{g['roc_auc']:.3f}",
                            "Risk ECE": f"{g['ece']:.3f}"} for name, g in summary["groups"].items()])
    cost = pd.DataFrame([{"Measure": {"mae": "Effort MAE", "brier": "Risk Brier", "roc_auc": "Risk ROC-AUC"}[k],
                          "Unseen minus seen": f"{c['difference']:+.4f}", "95% interval":
                          f"{c['low']:+.4f} to {c['high']:+.4f}", "p": f"{c['p']:.3f}"}
                         for k, c in summary["cost_of_unseen"].items()])
    projects = pd.DataFrame(summary["per_project"]).round(3).sort_values("test stories", ascending=False)
    return "\n".join([
        "# Predictions for projects the models never saw", "",
        "Generated by `erp-unseen-projects` (`erp/arena/unseen.py`). Leave one project out: for each of the "
        f"{summary['projects']} TAWOS projects with test stories, `{summary['config']}` (the router's configuration, "
        "its tuned settings) was trained without the project, whose key it then did not know (as for a platform "
        "project), and predicted all its stories; the same configuration trained with the project predicted its "
        "test stories. Evidence for the confidence score's unseen-project penalty "
        f"(x{PENALTIES['unseen_project']}) and cold-start penalty (x{PENALTIES['cold_start']}, fewer than "
        f"{COLD_START_SPRINTS} closed sprints).", "",
        md_table(groups), "",
        "The cost of being unseen, on the same test stories (paired bootstrap; for MAE and Brier, positive means "
        "worse when unseen):", "",
        md_table(cost), "",
        "How much of the seen models' accuracy each group keeps (1 = as good as seen). A data-support penalty in "
        "the confidence score (a factor on S) of about this size matches the accuracy lost:", "",
        md_table(pd.DataFrame([{"Group": name, **{k: f"{v:.2f}" for k, v in s.items()}}
                               for name, s in summary["support"].items()])), "",
        "Per project (test stories):", "",
        md_table(projects), "",
        "The FastText encoder was fitted on every project's training stories, the held-out one's included, so the "
        "text side is slightly optimistic for unseen projects. Cold and warm stories are different stories (a "
        "project's first sprints against its later ones), so their gap mixes history with how the project changed.",
    ]) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from saved predictions")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = OUT_DIR / "predictions.parquet"
    if args.report_only:
        stories = pd.read_parquet(saved)
    else:
        stories = run(ArenaData.load())
        stories.to_parquet(saved, index=False)
    summary = summarise(stories)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    args.report.write_text(build_report(summary), encoding="utf-8")
    train.log(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
