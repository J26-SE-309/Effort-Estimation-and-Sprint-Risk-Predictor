"""Phase 3: train the deployable bundle - M1 and M2 with C1 calibration, C2 intervals and X1 explanations.

Trains SBERT + LightGBM for effort (M1) and risk (M2) on the oldest 60% of each project's sprints (as in
Phase 2), fits the calibrator (C1) and the conformal intervals (C2) on the next 20%, chooses the risk
threshold there, and scores everything once on the newest 20%. Writes:
  ml-engine/models/<name>/                  the bundle (committed with the code; see bundle.py)
  reports/uncertainty-and-explanations.md   calibration, interval coverage and explanations on the test split
The bundle is reloaded from disk and must reproduce the in-memory predictions before the report is written.

Usage:
    erp-train-bundle                        # writes ml-engine/models/sbert-lightgbm-v1/
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from erp import config
from erp.explore.profile_tawos import md_table, pct
from erp.features import catalog
from erp.models import calibration, encoders, explain, first_models, metrics
from erp.models.bundle import INTERVAL_COVERAGES, RISK_BANDS, ModelBundle, code_version
from erp.models.inputs import design

NAME = "sbert-lightgbm-v1"


def train(frame: pd.DataFrame, text: np.ndarray) -> tuple[ModelBundle, dict]:
    parts = frame["split"]
    cal = (parts == "cal").to_numpy()
    x1, y1 = design(frame, text, "effort"), np.log1p(frame["story_points"])
    m1 = first_models.fit("effort", x1, y1, parts)
    intervals = calibration.ConformalIntervals().fit(y1[cal], m1.predict(x1[cal]), INTERVAL_COVERAGES)

    x2, y2 = design(frame, text, "risk"), frame["at_risk"].astype(int)
    m2 = first_models.fit("risk", x2, y2, parts)
    raw = m2.predict_proba(x2)[:, 1]
    calibrator = calibration.Calibrator("auto").fit(raw[cal], y2[cal])
    platt = calibration.Calibrator("platt").fit(raw[cal], y2[cal])
    threshold = metrics.choose_threshold(y2[cal], calibrator.predict(raw[cal]))

    bundle = ModelBundle(m1=m1.booster_, m2=m2.booster_, calibrator=calibrator, intervals=intervals,
                         threshold=threshold, bands=dict(RISK_BANDS))
    extras = {"raw": pd.Series(raw, index=frame.index), "platt": platt,
              "raw_threshold": metrics.choose_threshold(y2[cal], raw[cal])}
    return bundle, extras


def manifest(frame: pd.DataFrame, results: dict) -> dict:
    encoder = encoders.SbertEncoder
    return {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "encoder": {"name": encoder.name, "model_id": encoder.model_id, "revision": encoder.revision,
                    "dimensions": encoder.dimensions, "text": encoder.text},
        "learner": {"name": "LightGBM", "params": first_models.LGBM,
                    "early_stopping_rounds": first_models.EARLY_STOPPING},
        "categorical": {c: sorted(frame[c].astype(str).unique().tolist()) for c in ("issue_type", "priority_level",
                                                                                   "project_key")},
        "data": {"source": "TAWOS v1.1", "stories": len(frame), "split": frame["split"].value_counts().to_dict(),
                 "split_rule": "per project by sprint start: oldest 60% train, next 20% calibration, newest 20% test",
                 "risk_labels": "automatic R1-R6, before the 200-story human check"},
        "test_metrics": results,
    }


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    which = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    frame = pd.DataFrame({"bin": which, "p": p, "y": np.asarray(y, float)})
    table = frame.groupby("bin").agg(stories=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    table.index = [f"{b / bins:.1f}–{(b + 1) / bins:.1f}" for b in table.index]
    return table


def build_report(frame, bundle, extras, predictions, round_trip, directory) -> tuple[str, dict]:
    test = frame[frame["split"] == "test"]
    pred = predictions.loc[test.index]
    y = test["at_risk"].astype(int)
    raw = extras["raw"][test.index]

    before = metrics.classification_report(y, raw, extras["raw_threshold"])
    after = metrics.classification_report(y, pred["spillover_probability"], bundle.threshold)
    platt = metrics.classification_report(y, extras["platt"].predict(raw),
                                          metrics.choose_threshold(frame.loc[frame["split"] == "cal", "at_risk"],
                                                                   extras["platt"].predict(
                                                                       extras["raw"][frame["split"] == "cal"])))
    effort = metrics.effort_report(test["story_points"], pred["predicted_story_points"], test["Project_ID"])
    coverage = {c: calibration.coverage_and_width(test["story_points"], pred[f"interval_{c:g}_low"],
                                                  pred[f"interval_{c:g}_high"]) for c in INTERVAL_COVERAGES}
    results = {"m1": {**effort, **{f"coverage_{c:g}": v[0] for c, v in coverage.items()},
                      **{f"mean_width_{c:g}": v[1] for c, v in coverage.items()}},
               "m2": {"before_calibration": before, "after_calibration": after}}

    fmt3 = lambda v: f"{v:.3f}"  # noqa: E731
    cal_table = pd.DataFrame([
        {"Scores": "Raw LightGBM scores", **{k: fmt3(v) for k, v in before.items()}},
        {"Scores": f"**Calibrated ({bundle.calibrator.method}, in the bundle)**",
         **{k: fmt3(v) for k, v in after.items()}},
        {"Scores": "Calibrated (Platt, for comparison only)", **{k: fmt3(v) for k, v in platt.items()}},
    ])[["Scores", "brier", "ece", "roc_auc", "precision", "recall", "f1", "threshold"]].rename(columns={
        "brier": "Brier", "ece": "ECE", "roc_auc": "ROC-AUC", "precision": "Precision", "recall": "Recall",
        "f1": "F1", "threshold": "Threshold"})
    rel = reliability(y, pred["spillover_probability"])
    rel_table = pd.DataFrame({"Calibrated probability": rel.index, "Stories": rel["stories"].astype(int),
                              "Mean calibrated": rel["predicted"].map(fmt3),
                              "Observed at-risk rate": rel["observed"].map(fmt3)})
    levels = pred.assign(y=y).groupby("risk_level")["y"].agg(["size", "mean"]).reindex(["low", "medium", "high"])
    level_table = pd.DataFrame({"Risk level": levels.index, "Test stories": levels["size"].fillna(0).astype(int),
                                "Share": [pct(n, len(test)) for n in levels["size"].fillna(0)],
                                "Actually at risk": levels["mean"].map(lambda v: "–" if pd.isna(v) else f"{v:.0%}")})

    per_project = test.assign(low=pred["interval_0.8_low"], high=pred["interval_0.8_high"]).groupby("Project_ID")
    project_cov = per_project.apply(lambda g: pd.Series({
        "n": len(g), "cov": ((g["story_points"] >= g["low"]) & (g["story_points"] <= g["high"])).mean()}),
        include_groups=False)
    project_cov = project_cov[project_cov["n"] >= 30]["cov"]
    interval_table = pd.DataFrame([{
        "Interval": f"{c:.0%}", "Coverage on test": f"{coverage[c][0]:.1%}",
        "Mean width (points)": f"{coverage[c][1]:.1f}",
        "q (log scale)": f"{bundle.intervals.quantiles[f'{c:g}']:.3f}",
        "Example: M1 says 3 points": "{:.1f}–{:.1f}".format(*[float(v[0]) for v in bundle.intervals.interval(
            [np.log1p(3.0)], c)]),
    } for c in INTERVAL_COVERAGES])

    x1 = design(test, np.stack(test["_text"].to_numpy()), "effort")[bundle.m1.feature_name()]
    x2 = design(test, np.stack(test["_text"].to_numpy()), "risk")[bundle.m2.feature_name()]
    effort_imp = explain.global_importance(explain.by_factor(explain.contributions(bundle.m1, x1), catalog.factor_of))
    risk_imp = explain.global_importance(explain.by_factor(explain.contributions(bundle.m2, x2), catalog.factor_of))
    imp_table = pd.DataFrame({"Planning factor": effort_imp.index.union(risk_imp.index)})
    imp_table["Effort (M1)"] = imp_table["Planning factor"].map(effort_imp).map(lambda v: f"{v:.0%}")
    imp_table["Risk (M2)"] = imp_table["Planning factor"].map(risk_imp).map(lambda v: f"{v:.0%}")
    imp_table = imp_table.sort_values("Risk (M2)", key=lambda s: s.str.rstrip("%").astype(float), ascending=False)

    sample = test[test["project_key"] == "MESOS"].sort_index()
    sample = pd.concat([sample[sample["at_risk"]].head(3), sample[~sample["at_risk"]].head(2)])
    examples = pd.DataFrame({
        "Story": sample["Issue_Key"], "Points (actual)": sample["story_points"].map("{:g}".format),
        "M1 (80% interval)": [f"{p:.1f} ({lo:.1f}–{hi:.1f})" for p, lo, hi in zip(
            pred.loc[sample.index, "predicted_story_points"], pred.loc[sample.index, "interval_0.8_low"],
            pred.loc[sample.index, "interval_0.8_high"], strict=True)],
        "At risk (actual)": np.where(sample["at_risk"], "yes", "no"),
        "M2 probability / level": [f"{p:.2f} / {lvl}" for p, lvl in zip(
            pred.loc[sample.index, "spillover_probability"], pred.loc[sample.index, "risk_level"], strict=True)],
        "Top reasons for risk": ["; ".join(f"{r['factor']} ({r['share']:.0%})" for r in reasons) or "–"
                                 for reasons in pred.loc[sample.index, "risk_reasons"]],
    })

    files = ", ".join(f"`{p.name}` ({p.stat().st_size / 1e6:.1f} MB)" for p in sorted(directory.iterdir()))
    report = "\n".join([
        "# Uncertainty and explanations (Phase 3)", "",
        "Generated by `erp-train-bundle`. ML guide Phase 3: the calibrator C1 for risk, conformal intervals C2 for "
        "effort and SHAP explanations X1 grouped into planning factors, all fitted on the calibration split and "
        "scored once on the test split (the newest 20% of each project's sprints).", "",
        f"**Bundle:** `ml-engine/models/{NAME}/`: {files}. Reloaded from disk it reproduces the in-memory "
        f"predictions (largest difference {round_trip:.1e}).", "",
        "## 1. C1: honest risk probabilities", "",
        f"Isotonic regression was chosen by the ML guide's rule ({len(frame[frame['split'] == 'cal']):,} calibration "
        f"stories, more than {calibration.ISOTONIC_MIN_SAMPLES:,}), before looking at the test split; Platt scaling is "
        f"shown for comparison only. NFR3 asks for ECE ≤ 0.10: {after['ece']:.3f} after calibration "
        f"({'met' if after['ece'] <= 0.10 else 'not met'}). The threshold is chosen on the calibration split for the "
        f"best F1 with recall ≥ {metrics.MIN_RECALL:.2f}.", "",
        (f"The raw LightGBM scores were already well calibrated (ECE {before['ece']:.3f}), so C1 changes little here: "
         f"ECE {before['ece']:.3f} → {after['ece']:.3f}, Brier {before['brier']:.3f} → {after['brier']:.3f}. "
         "It stays in the bundle because the ML guide requires it for every risk model: other learners (SVM, "
         "stacking, M3) give scores that are not probabilities at all, and C1 must be refitted whenever M2 is "
         "retrained."
         if abs(after["ece"] - before["ece"]) < 0.02 else
         f"C1 moves ECE from {before['ece']:.3f} to {after['ece']:.3f} and Brier from {before['brier']:.3f} to "
         f"{after['brier']:.3f}."), "",
        md_table(cal_table), "",
        "Reliability on the test split (a calibrated model has *Mean calibrated* close to *Observed*):", "",
        md_table(rel_table), "",
        f"Risk levels use the ML guide's bands (low < {RISK_BANDS['medium']}, medium {RISK_BANDS['medium']}–"
        f"{RISK_BANDS['high']}, high > {RISK_BANDS['high']}):", "",
        md_table(level_table), "",
        "## 2. C2: effort ranges that keep their promise", "",
        f"M1 on the test split: MAE {effort['mae']:.2f} points, SA {effort['sa']:.1f} (as in Phase 2). Split "
        "conformal on the log scale: the range is multiplicative, so it is wider for big stories. Coverage is the "
        "share of test stories whose actual points fall inside their range (NFR2).", "",
        md_table(interval_table), "",
        f"Per project (at least 30 test stories), the 80% range covers between {project_cov.min():.0%} and "
        f"{project_cov.max():.0%} of stories: the pooled guarantee does not hold for every team. Calibrating per "
        "project (Mondrian conformal) is the fix if the thesis needs per-team coverage.", "",
        "## 3. X1: why the models predict what they do", "",
        "Mean absolute SHAP value per planning factor on the test split, as a share (the 384 SBERT values are summed "
        "into *Story size and content*):", "",
        md_table(imp_table), "",
        "Examples from the test split (MESOS, public Jira). Reasons are the factors pushing the risk up, with their "
        f"share of the total contribution (factors below {explain.MIN_REASON_SHARE:.0%} are not reasons):", "",
        md_table(examples), "",
        "The planted-signal test (ML guide 7.2) is `tests/test_explain.py`: on made-up data where only two features "
        "matter, the same training and explanation code must rank those two first and name them as the first reason "
        "for the risky stories. It passes and runs in CI. It also showed that with the untuned Phase 2 settings a "
        "pure-noise feature still gets about a quarter of the SHAP weight, because the trees fit some of the random "
        "part of the labels. The explanations report that faithfully; reducing it is a job for the Phase 4 tuning.",
        "",
        "## 4. Still to come", "",
        "- The risk labels wait for the 200-story human check; the bundle is retrained after it.",
        "- `confidence_score` (ML guide 4.6) needs its formula agreed with the supervisor; it will be added to the "
        "bundle's predictions then.",
        "- The recommendation engine (R2) turns the top reasons into actions in Phase 5.", "",
    ])
    return report, results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path,
                        default=config.REPO_ROOT / "ml-engine" / "reports" / "uncertainty-and-explanations.md")
    parser.add_argument("--out", type=Path, default=config.MODEL_BUNDLES_DIR / NAME)
    args = parser.parse_args(argv)

    frame = first_models.load_dataset(config.INTERIM_DIR)
    snapshot = pd.read_parquet(config.INTERIM_DIR / "snapshot.parquet", columns=["Issue_ID", "Issue_Key"])
    frame = frame.join(snapshot.set_index("Issue_ID"))
    text = encoders.SbertEncoder().transform(encoders.story_text(frame))
    frame["_text"] = list(text)

    bundle, extras = train(frame, text)
    predictions = bundle.predict(frame, text)
    bundle.manifest = manifest(frame, {})
    bundle.save(args.out)

    test = frame[frame["split"] == "test"]
    reloaded = ModelBundle.load(args.out).predict(test, np.stack(test["_text"].to_numpy()), explain_top=0)
    numeric = ["predicted_story_points", "spillover_probability", "interval_0.8_low", "interval_0.9_high"]
    round_trip = float((reloaded[numeric] - predictions.loc[test.index, numeric]).abs().max().max())
    if round_trip > 1e-9:
        raise SystemExit(f"The saved bundle does not reproduce the predictions (difference {round_trip}).")

    report, results = build_report(frame, bundle, extras, predictions, round_trip, args.out)
    bundle.manifest = manifest(frame, results)
    bundle.save(args.out)
    args.report.write_text(report, encoding="utf-8")
    print(f"Wrote the bundle to {args.out} and {args.report}")


if __name__ == "__main__":
    main()
