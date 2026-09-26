"""Phase 2: the first honest models (ML guide section 12).

Correctly computed baselines, then SBERT + LightGBM for effort (M1) and risk (M2), trained on the oldest 60%
of each project's sprints, tuned (early stopping, risk threshold) on the next 20% and scored once on the most
recent 20%. Two reduced versions of each model, text only and structured features only, show what each half
contributes. Writes:
  Datasets/effort-risk/models/first/   m1.txt and m2.txt (LightGBM) and meta.json
  reports/first-models.md              the results

Usage:
    erp-train-first                         # writes ml-engine/reports/first-models.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from erp import config
from erp.explore.profile_tawos import md_table
from erp.features import catalog
from erp.models import encoders, metrics, split

CATEGORICAL = ["issue_type", "priority_level", "project_key"]
SEED = 42
LGBM = {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.5,
        "subsample": 0.8, "subsample_freq": 1, "n_estimators": 3000, "random_state": SEED, "verbose": -1}
EARLY_STOPPING = 100
MODELS_DIR = config.WORK_DIR / "models" / "first"


def load_dataset(interim: Path) -> pd.DataFrame:
    snapshot = pd.read_parquet(interim / "snapshot.parquet")
    features = pd.read_parquet(interim / "features.parquet")
    labels = pd.read_parquet(interim / "labels.parquet")[["Issue_ID", "at_risk", "r1", "r2", "r3", "r4", "r5", "r6"]]
    keep = ["Issue_ID", "Project_ID", "Sprint_ID", "sprint_start", "title", "description_text"]
    frame = snapshot[keep].merge(features, on="Issue_ID").merge(labels, on="Issue_ID").set_index("Issue_ID")
    frame["split"] = split.temporal_split(frame)
    return frame


def structured(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    """The catalogue features as model inputs. M1 never sees story_points: they are its answer."""
    names = [n for n in catalog.names() if not (task == "effort" and n == "story_points")]
    x = frame[names].copy()
    for column in x.columns:
        if column in CATEGORICAL:
            x[column] = pd.Categorical(x[column].astype(str))
        elif pd.api.types.is_bool_dtype(x[column]):
            x[column] = x[column].astype(int)
    return x


def design(frame: pd.DataFrame, text: np.ndarray, task: str, use_text: bool, use_features: bool) -> pd.DataFrame:
    parts = []
    if use_text:
        parts.append(pd.DataFrame(text, index=frame.index, columns=encoders.SbertEncoder().columns()))
    if use_features:
        parts.append(structured(frame, task))
    return pd.concat(parts, axis=1)


def fit(task: str, x: pd.DataFrame, y: pd.Series, parts: pd.Series):
    import lightgbm as lgb

    model = (lgb.LGBMRegressor(objective="l1", **LGBM) if task == "effort"
             else lgb.LGBMClassifier(objective="binary", **LGBM))
    train, cal = parts == "train", parts == "cal"
    model.fit(x[train], y[train], eval_X=(x[cal],), eval_y=(y[cal],),
              callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    return model


def effort_results(frame: pd.DataFrame, text: np.ndarray) -> tuple[pd.DataFrame, dict, object, pd.Series]:
    parts, points = frame["split"], frame["story_points"]
    test = frame[parts == "test"]
    train_points = frame[parts == "train"].groupby("Project_ID")["story_points"]
    guesses = {
        "Mean of the project's training stories": test["Project_ID"].map(train_points.mean()),
        "Median of the project's training stories": test["Project_ID"].map(train_points.median()),
    }
    rows = [{"Method": "Random guessing (reference for SA)", "MAE": metrics.random_guessing_mae(
        test["story_points"], test["Project_ID"]), "MdAE": np.nan, "SA": 0.0}]
    for name, guess in guesses.items():
        report = metrics.effort_report(test["story_points"], guess, test["Project_ID"])
        rows.append({"Method": name, "MAE": report["mae"], "MdAE": report["mdae"], "SA": report["sa"]})
    target = np.log1p(points)
    main, main_pred = None, None
    for name, use_text, use_features in [("SBERT + LightGBM, text only", True, False),
                                         ("LightGBM on features only (no text)", False, True),
                                         ("**SBERT + LightGBM (M1)**", True, True)]:
        x = design(frame, text, "effort", use_text, use_features)
        model = fit("effort", x, target, parts)
        predicted = np.expm1(model.predict(x[parts == "test"]))
        report = metrics.effort_report(test["story_points"], predicted, test["Project_ID"])
        rows.append({"Method": name, "MAE": report["mae"], "MdAE": report["mdae"], "SA": report["sa"]})
        if use_text and use_features:
            main, main_pred = model, pd.Series(predicted, index=test.index)
    table = pd.DataFrame(rows)
    per_project = test.assign(predicted=main_pred, median=guesses["Median of the project's training stories"])
    p_value = metrics.wilcoxon_p(np.abs(test["story_points"] - main_pred),
                                 np.abs(test["story_points"] - per_project["median"]))
    return table, {"best_iteration": int(main.best_iteration_ or 0), "p_vs_median": p_value}, main, per_project


def risk_results(frame: pd.DataFrame, text: np.ndarray) -> tuple[pd.DataFrame, dict, object, pd.DataFrame]:
    parts, y = frame["split"], frame["at_risk"].astype(int)
    cal, test = frame[parts == "cal"], frame[parts == "test"]
    prior = float(y[parts == "train"].mean())
    rows = []

    def add(name: str, cal_scores, test_scores, fixed_threshold: float | None = None):
        threshold = fixed_threshold if fixed_threshold is not None else metrics.choose_threshold(
            cal["at_risk"], cal_scores)
        report = metrics.classification_report(test["at_risk"], test_scores, threshold)
        rows.append({"Method": name, **report})

    majority = 1.0 if prior >= 0.5 else 0.0
    add(f"Majority class ({'at risk' if majority else 'not at risk'} for every story)",
        np.full(len(cal), prior), np.full(len(test), prior), fixed_threshold=prior + (1e-9 if not majority else -1e-9))
    history_rate = frame["historical_spillover_rate"].fillna(prior)
    add("Team's recent spillover rate as the probability", history_rate[parts == "cal"], history_rate[parts == "test"])
    heuristic = history_rate[parts == "test"]
    main, main_scores, threshold = None, None, None
    for name, use_text, use_features in [("SBERT + LightGBM, text only", True, False),
                                         ("LightGBM on features only (no text)", False, True),
                                         ("**SBERT + LightGBM (M2)**", True, True)]:
        x = design(frame, text, "risk", use_text, use_features)
        model = fit("risk", x, y, parts)
        cal_scores = model.predict_proba(x[parts == "cal"])[:, 1]
        test_scores = model.predict_proba(x[parts == "test"])[:, 1]
        add(name, cal_scores, test_scores)
        if use_text and use_features:
            main, main_scores, threshold = model, pd.Series(test_scores, index=test.index), rows[-1]["threshold"]
    table = pd.DataFrame(rows)
    per_project = test.assign(score=main_scores, heuristic=heuristic)
    difference = metrics.auc_difference(test["at_risk"], main_scores, heuristic)
    return table, {"threshold": threshold, "prior": prior, "best_iteration": int(main.best_iteration_ or 0),
                   "auc_vs_heuristic": difference}, main, per_project


def importance(model) -> pd.Series:
    """Gain importance, with the 384 SBERT values summed into one 'story text' factor (ML guide 4.7)."""
    gains = pd.Series(model.booster_.feature_importance("gain"), index=model.booster_.feature_name())
    group_of = {f.name: f.group for f in catalog.FEATURES}
    groups = gains.groupby(lambda n: "story text (SBERT)" if n.startswith("sbert_") else group_of.get(n, n)).sum()
    return (groups / groups.sum()).sort_values(ascending=False)


def build_report(frame, effort_table, risk_table, effort_imp, risk_imp, effort_projects, risk_projects, meta) -> str:
    sizes = frame["split"].value_counts().reindex(["train", "cal", "test"])
    dates = frame.groupby("split")["sprint_start"].agg(["min", "max"]).reindex(["train", "cal", "test"])
    fmt = lambda v, d=2: "–" if pd.isna(v) else f"{v:,.{d}f}"  # noqa: E731
    effort = effort_table.assign(**{c: effort_table[c].map(fmt) for c in ["MAE", "MdAE", "SA"]})
    risk = risk_table.assign(**{c: risk_table[c].map(lambda v: fmt(v, 3)) for c in
                                ["precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "ece", "threshold"]})
    risk = risk.rename(columns={"precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
                                "pr_auc": "PR-AUC", "brier": "Brier", "ece": "ECE", "threshold": "Threshold"})
    median_row = effort_table[effort_table["Method"].str.startswith("Median")].iloc[0]
    m1 = effort_table.iloc[-1]
    m2 = risk_table.iloc[-1]
    history_row = risk_table[risk_table["Method"].str.startswith("Team")].iloc[0]

    per_effort = effort_projects.groupby("Project_ID").apply(lambda g: pd.Series({
        "Stories": len(g), "MAE M1": metrics.mae(g["story_points"], g["predicted"]),
        "MAE median": metrics.mae(g["story_points"], g["median"])}), include_groups=False)
    per_risk = risk_projects.groupby("Project_ID").apply(lambda g: pd.Series({
        "At risk": g["at_risk"].mean(),
        "F1 M2": metrics.classification_report(g["at_risk"], g["score"], meta["risk"]["threshold"])["f1"]}),
        include_groups=False)
    keys = frame.groupby("Project_ID")["project_key"].first()
    per = per_effort.join(per_risk)
    per = per[per["Stories"] >= 30].sort_values("Stories", ascending=False)
    per_table = pd.DataFrame({
        "Project": per.index.map(keys), "Test stories": per["Stories"].astype(int),
        "MAE M1": per["MAE M1"].map(fmt), "MAE median guess": per["MAE median"].map(fmt),
        "At risk": per["At risk"].map(lambda v: f"{v:.0%}"), "F1 M2": per["F1 M2"].map(lambda v: fmt(v, 3)),
    })
    imp = lambda s: ", ".join(f"{k} {v:.0%}" for k, v in s.head(5).items())  # noqa: E731

    return "\n".join([
        "# First models (Phase 2)", "",
        "Generated by `erp-train-first`. The first honest models of ML guide Phase 2: correctly computed baselines, "
        "then SBERT (E3) + LightGBM for effort (M1) and risk (M2). No hyperparameter search yet (that is the "
        "Phase 4 arena); LightGBM defaults with early stopping on the calibration split.", "",
        "## 1. Data and split", "",
        "Each project's sprints are ordered by start date: the oldest 60% train, the next 20% calibrate "
        "(early stopping, the risk threshold), the newest 20% are the test set, used once for the numbers below.", "",
        md_table(pd.DataFrame({"Split": sizes.index, "Stories": sizes.map("{:,}".format).to_numpy(),
                               "Sprints from": dates["min"].dt.strftime("%Y-%m").to_numpy(),
                               "to": dates["max"].dt.strftime("%Y-%m").to_numpy()})), "",
        "## 2. Effort (M1): story points", "",
        "MAE and MdAE in story points on each project's own scale; SA against random guessing (another issue's "
        "actual points from the same project), so 0 means no better than guessing.", "",
        md_table(effort), "",
        f"M1 is off by {m1['MAE']:.2f} points on average, against {median_row['MAE']:.2f} for always guessing the "
        f"project's median ({(1 - m1['MAE'] / median_row['MAE']):.0%} lower; Wilcoxon signed-rank test on the paired "
        f"errors p = {meta['effort_p']:.2g}). What drives it: {imp(effort_imp)}.", "",
        "## 3. Risk (M2): will the story run into trouble?", "",
        f"The threshold is chosen on the calibration split for the best F1 with recall ≥ {metrics.MIN_RECALL:.2f} "
        f"(NFR). Scores are raw (C1 calibration comes in Phase 3), so Brier and ECE are 'before calibration'. "
        f"Test stories at risk: {risk_projects['at_risk'].mean():.1%}.", "",
        md_table(risk), "",
        f"M2 reaches ROC-AUC {m2['roc_auc']:.3f} and F1 {m2['f1']:.3f}, against {history_row['roc_auc']:.3f} and "
        f"{history_row['f1']:.3f} for the team's recent spillover rate alone (ROC-AUC difference "
        f"{meta['auc'][0]:+.3f}, 95% bootstrap interval {meta['auc'][1]:+.3f} to {meta['auc'][2]:+.3f}). With half "
        "the test stories at risk, flagging nearly everything already gives F1 ≈ 0.67, so ROC-AUC and PR-AUC show "
        f"the ranking skill better than F1. What drives it: {imp(risk_imp)}. The 384 SBERT values share many "
        "splits between them, so their share of the gain overstates their value: the text-only model is weak for "
        "risk.", "",
        "## 4. Per project (test split, projects with at least 30 test stories)", "",
        md_table(per_table), "",
        "## 5. Before reading too much into this", "",
        "- The risk labels are automatic and still wait for the 200-story human check; M2 is retrained after it.",
        "- The quality, acceptance-criteria and traceability features are proxies for Components 1–3.",
        "- One pooled model for all projects; per-project models and the full encoder × learner arena are Phase 4.",
        "- The test split is the newest sprints of every project, so these numbers include some drift over time.",
        "",
    ])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=config.REPO_ROOT / "ml-engine" / "reports" / "first-models.md")
    parser.add_argument("--interim", type=Path, default=config.INTERIM_DIR)
    parser.add_argument("--models", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    frame = load_dataset(args.interim)
    text = encoders.SbertEncoder().fit(encoders.story_text(frame)).transform(encoders.story_text(frame))
    effort_table, effort_meta, m1, effort_projects = effort_results(frame, text)
    risk_table, risk_meta, m2, risk_projects = risk_results(frame, text)

    args.models.mkdir(parents=True, exist_ok=True)
    m1.booster_.save_model(args.models / "m1.txt")
    m2.booster_.save_model(args.models / "m2.txt")
    meta = {"encoder": encoders.SbertEncoder.model_id, "learner": "LightGBM", "params": LGBM,
            "effort": effort_meta, "risk": risk_meta, "split": frame["split"].value_counts().to_dict(),
            "features": catalog.names(), "categorical": CATEGORICAL}
    (args.models / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    args.report.write_text(build_report(frame, effort_table, risk_table, importance(m1),
                                        importance(m2), effort_projects, risk_projects,
                                        {"risk": risk_meta, "effort_p": effort_meta["p_vs_median"],
                                         "auc": risk_meta["auc_vs_heuristic"]}), encoding="utf-8")
    print(f"Wrote {args.report} and the models to {args.models}")


if __name__ == "__main__":
    main()
