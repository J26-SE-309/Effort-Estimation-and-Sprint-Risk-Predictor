"""Adaptive effort intervals (C2) and the confidence score for every arena configuration.

For each saved configuration: a difficulty model is trained on the effort model's inner-fold predictions
(training stories only), the adaptive intervals are calibrated on the calibration split, and the confidence
score's reference widths are taken from the calibration split too. model.json keeps the plain split-conformal
intervals as intervals_split for comparison. The reloaded configuration must reproduce the new intervals
exactly. The test split then answers two questions, written to reports/confidence.md:
  - do the adaptive intervals keep their coverage, and do they follow the real difficulty of a story?
  - do the confidence bands order the accuracy (High better than Medium, Medium better than Low)?

Usage:
    erp-fit-intervals            # every saved configuration, then the report
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from erp import config as paths
from erp.arena import configs, predictor
from erp.arena.data import MODELS_DIR, WORK_DIR, ArenaData, targets
from erp.arena.train import log
from erp.explore.profile_tawos import md_table
from erp.models import confidence as conf
from erp.models.bundle import INTERVAL_COVERAGES
from erp.models.calibration import AdaptiveIntervals, Calibrator, ConformalIntervals, coverage_and_width

REPORT = paths.REPO_ROOT / "ml-engine" / "reports" / "confidence.md"


def inner_fold_effort(name: str, manifest: dict) -> pd.Series | None:
    """The effort model's inner-fold predictions (log points) for training stories, or None if not kept."""
    if name == "stack":
        bases = sorted(set(manifest["effort"]["bases"]) | set(manifest["risk"]["bases"]))
        oof = {b: pd.read_parquet(WORK_DIR / "oof" / f"{b}.parquet") for b in bases}
        rows = oof[bases[0]].index
        log_points, _ = predictor.combine(manifest, {b: (oof[b].loc[rows, "effort_log"].to_numpy(),
                                                         oof[b].loc[rows, "risk_raw"].to_numpy()) for b in bases})
        return pd.Series(log_points, index=rows)
    path = WORK_DIR / "oof" / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)["effort_log"]
    path = WORK_DIR / "oof" / f"{name}-effort.parquet"  # DistilBERT keeps one file per task
    return pd.read_parquet(path)["prediction"] if path.exists() else None


def fit(arena: ArenaData, name: str) -> None:
    directory = MODELS_DIR / name
    path = directory / predictor.MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split = manifest["effort"].get("intervals_split", manifest["effort"]["intervals"])
    stored = pd.read_parquet(WORK_DIR / "predictions" / f"{name}.parquet").loc[arena.frame.index]
    log_points = stored["effort_log"].to_numpy()
    y = targets(arena.frame, "effort")
    cal = arena.mask("cal")
    oof = inner_fold_effort(name, manifest)
    if oof is not None:
        intervals = AdaptiveIntervals(arena.levels).fit_difficulty(
            arena.frame.loc[oof.index], oof.to_numpy(), y[arena.frame.index.get_indexer(oof.index)])
        intervals.fit(arena.frame[cal], y[cal], log_points[cal], INTERVAL_COVERAGES)
        spec = intervals.save(directory)
    else:
        intervals, spec = ConformalIntervals.from_dict(split), split
    low, high = intervals.bounds(arena.frame[cal], log_points[cal], 0.8)
    confidence = conf.Confidence().fit(conf.relative_width(low, high, np.expm1(log_points[cal])),
                                       arena.frame.loc[arena.mask("train"), "project_key"].unique())
    manifest.setdefault("levels", arena.levels)  # the stack's manifest had none
    manifest["effort"]["intervals"] = spec
    manifest["effort"]["intervals_split"] = split
    manifest["confidence"] = confidence.to_dict()
    path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")

    # The reloaded C2 must give exactly the same intervals for the same predictions. (The models' own
    # predictions were checked when they were trained; a network's last digits can depend on the batch size.)
    test = arena.mask("test")
    loaded = predictor.load(directory)
    for coverage in INTERVAL_COVERAGES:
        expected = intervals.bounds(arena.frame[test], log_points[test], coverage)
        got = loaded.intervals.bounds(arena.frame[test], log_points[test], coverage)
        if not all(np.array_equal(a, b) for a, b in zip(got, expected, strict=True)):
            raise AssertionError(f"{name}: the reloaded intervals differ")
    out = loaded.predict(arena.frame[test], test_text(arena, loaded))  # and the whole path runs end to end
    if out["confidence"].isna().any():
        raise AssertionError(f"{name}: predictions without a confidence band")
    log(f"{name}: {'adaptive' if oof is not None else 'split conformal'} intervals and confidence saved")


def test_text(arena: ArenaData, loaded):
    test = arena.mask("test")
    if isinstance(loaded, predictor.StackPredictor):
        return {b.encoder_name: arena.full_text(b.encoder_name)[test] for b in loaded.bases.values()}
    if loaded.encoder_name == "distilbert":
        return None
    return arena.full_text(loaded.encoder_name)[test]


# ---------------------------------------------------------------- evaluation on the test split


def evaluate(arena: ArenaData, name: str) -> dict:
    manifest = json.loads((MODELS_DIR / name / predictor.MANIFEST).read_text(encoding="utf-8"))
    test = arena.mask("test")
    stories = arena.frame[test]
    stored = pd.read_parquet(WORK_DIR / "predictions" / f"{name}.parquet").loc[stories.index]
    log_points, points = stored["effort_log"].to_numpy(), np.expm1(stored["effort_log"].to_numpy())
    actual = stories["story_points"].to_numpy(float)
    loaded = predictor.load(MODELS_DIR / name)
    split = ConformalIntervals.from_dict(manifest["effort"]["intervals_split"])
    result = {"name": name, "adaptive": manifest["effort"]["intervals"].get("method") == AdaptiveIntervals.method}
    for label, intervals in (("split", split), ("adaptive", loaded.intervals)):
        for coverage in INTERVAL_COVERAGES:
            low, high = intervals.bounds(stories, log_points, coverage)
            result[f"{label}_coverage_{coverage:g}"], result[f"{label}_width_{coverage:g}"] = coverage_and_width(
                actual, low, high)
    error = np.abs(np.log1p(actual) - log_points)
    if result["adaptive"]:
        difficulty = loaded.intervals.difficulty(stories, log_points)
        from scipy.stats import spearmanr

        result["difficulty_rho"] = float(spearmanr(difficulty, error).statistic)
        thirds = pd.qcut(pd.Series(difficulty).rank(method="first"), 3, labels=["easiest", "middle", "hardest"])
        for label, intervals in (("split", split), ("adaptive", loaded.intervals)):
            low, high = intervals.bounds(stories, log_points, 0.8)
            inside = pd.Series((actual >= low) & (actual <= high))
            result[f"{label}_coverage_by_third"] = inside.groupby(thirds.to_numpy(), observed=True).mean().to_dict()
    probability = Calibrator.from_dict(manifest["risk"]["calibrator"]).predict(stored["risk_raw"])
    low, high = loaded.intervals.bounds(stories, log_points, 0.8)
    scored = loaded.confidence.score(stories, low, high, points, probability)
    frame = scored.assign(abs_error=np.abs(actual - points), inside=(actual >= low) & (actual <= high),
                          width=high - low, brier=(probability - stories["at_risk"].to_numpy()) ** 2,
                          correct=(probability >= manifest["risk"]["threshold"]) == stories["at_risk"].to_numpy(),
                          cold=stories["history_sprints"].fillna(0).to_numpy() < conf.COLD_START_SPRINTS)
    bands = frame.groupby("confidence").agg(stories=("confidence", "size"), mae=("abs_error", "mean"),
                                            coverage=("inside", "mean"), width=("width", "mean"),
                                            brier=("brier", "mean"), correct=("correct", "mean"))
    bands = bands.reindex(["high", "medium", "low"]).dropna(how="all")
    result["bands"] = bands
    ordered = lambda column, better: bool(np.all(np.diff(bands[column].to_numpy()) * better > 0))  # noqa: E731
    result["ordered"] = {"MAE": ordered("mae", 1), "Brier": ordered("brier", 1), "risk correct": ordered(
        "correct", -1)}
    result["cold"] = frame.groupby("cold").agg(stories=("cold", "size"), mae=("abs_error", "mean"),
                                               brier=("brier", "mean"))
    from scipy.stats import spearmanr

    result["parts"] = {"effort certainty vs error": float(spearmanr(frame["effort_certainty"],
                                                                    frame["abs_error"]).statistic),
                       "risk certainty vs Brier": float(spearmanr(frame["risk_certainty"], frame["brier"]).statistic)}
    return result


def build_report(results: list[dict], winner: str | None) -> str:
    labels = {c.id: c.label for c in configs.CONFIGS}
    rows = []
    for r in results:
        rows.append({"Configuration": labels[r["name"]], "Intervals": "adaptive" if r["adaptive"] else "split only",
                     "80%: split": f"{r['split_coverage_0.8']:.1%} / {r['split_width_0.8']:.1f}",
                     "80%: adaptive": f"{r['adaptive_coverage_0.8']:.1%} / {r['adaptive_width_0.8']:.1f}",
                     "90%: split": f"{r['split_coverage_0.9']:.1%} / {r['split_width_0.9']:.1f}",
                     "90%: adaptive": f"{r['adaptive_coverage_0.9']:.1%} / {r['adaptive_width_0.9']:.1f}",
                     "Difficulty vs real error (Spearman)": f"{r['difficulty_rho']:.2f}" if "difficulty_rho" in r
                     else "–"})
    third_rows = []
    for r in results:
        if "adaptive_coverage_by_third" not in r:
            continue
        for label in ("split", "adaptive"):
            c = r[f"{label}_coverage_by_third"]
            third_rows.append({"Configuration": labels[r["name"]], "Intervals": label,
                               "Easiest third": f"{c['easiest']:.1%}", "Middle third": f"{c['middle']:.1%}",
                               "Hardest third": f"{c['hardest']:.1%}"})
    check = lambda ok: "yes" if ok else "**no**"  # noqa: E731
    order_rows = [{"Configuration": labels[r["name"]], "Stories high / medium / low": " / ".join(
        str(int(r["bands"]["stories"].get(b, 0))) for b in ("high", "medium", "low")),
        "MAE ordered": check(r["ordered"]["MAE"]), "Brier ordered": check(r["ordered"]["Brier"]),
        "Risk calls ordered": check(r["ordered"]["risk correct"]),
        "E vs error (Spearman)": f"{r['parts']['effort certainty vs error']:.2f}",
        "R vs Brier (Spearman)": f"{r['parts']['risk certainty vs Brier']:.2f}"} for r in results]
    main = next((r for r in results if r["name"] == winner), results[0])
    total = main["bands"]["stories"].sum()
    band_rows = [{"Band": band, "Stories": f"{int(b['stories']):,} ({b['stories'] / total:.0%})",
                  "MAE (points)": f"{b['mae']:.2f}", "80% interval covers": f"{b['coverage']:.1%}",
                  "Mean 80% width": f"{b['width']:.1f}", "Brier": f"{b['brier']:.3f}",
                  "Risk call right": f"{b['correct']:.1%}"} for band, b in main["bands"].iterrows()]
    cold = main["cold"]
    cold_rows = [{"Team history": "fewer than 3 closed sprints" if k else "3 or more", "Stories": int(v["stories"]),
                  "MAE (points)": f"{v['mae']:.2f}", "Brier": f"{v['brier']:.3f}"} for k, v in cold.iterrows()]
    return "\n".join([
        "# Adaptive intervals and the confidence score", "",
        "Generated by `erp-fit-intervals`. Test split only (the newest 20% of each project's sprints).", "",
        "## 1. Why the intervals had to change", "",
        "Plain split conformal (Phase 3) adds the same margin to every story on the log scale, so every story's "
        "80% range was the same multiple of its prediction (about 1.36 × (points + 1) wide): the width said how "
        "big a story was, not how sure the model was. Adaptive intervals (normalized split conformal) scale the "
        "margin by a difficulty model that predicts how far off the effort model tends to be for a story like this "
        "one. It learns from the effort model's inner-fold predictions on training stories (made by models that "
        "never saw them) and uses the story's features and predicted size; the calibration split then fixes the "
        "margin so the 80% and 90% guarantees hold as before.", "",
        "## 2. Coverage and width (coverage / mean width in points)", "",
        md_table(pd.DataFrame(rows)), "",
        "NFR3 asks for coverage within 5 points of the nominal level. The Spearman correlation shows whether the "
        "difficulty model ranks stories by how wrong the effort model really was on the test split (0 = no "
        "relation).", "",
        "Coverage of the 80% interval in thirds of the test stories, from easiest to hardest by the difficulty "
        "model. Plain intervals over-cover easy stories and under-cover hard ones; adaptive intervals should be "
        "close to 80% in all three:", "",
        md_table(pd.DataFrame(third_rows)), "",
        "## 3. The confidence score", "",
        "confidence_score = S × (E + R) / 2, shown as a band: High from 0.6, Medium from 0.35, Low below.", "",
        "- **E, effort certainty:** the share of calibration stories whose 80% interval was relatively wider than "
        "this one's.",
        "- **R, risk certainty:** |2p − 1| for the calibrated risk probability p (0 = coin flip).",
        "- **S, data support:** 1, × 0.7 for a team with fewer than 3 closed sprints, × 0.7 for a project the "
        "model never saw, × 0.85 for each unavailable upstream feature group (FR17).", "",
        f"The bands of {labels.get(main['name'], main['name'])} (the router's pooled winner) on the test split:", "",
        md_table(pd.DataFrame(band_rows)), "",
        "The bands are honest only if the accuracy is ordered: High better than Medium, Medium better than Low. "
        "For every configuration:", "",
        md_table(pd.DataFrame(order_rows)), "",
        "E vs error below 0 means stories with more effort certainty really had smaller errors; R vs Brier below 0 "
        "means firmer risk calls really were more often right.", "",
        "Does the cold-start penalty point the right way? The same configuration's test stories by team history:",
        "", md_table(pd.DataFrame(cold_rows)), "",
        "## 4. For the supervisor", "",
        "- The formula, the weights (E and R count equally), the penalties (0.7, 0.7, 0.85), the cold-start "
        "threshold (3 sprints) and the band limits (0.6, 0.35) are starting values to agree on.",
        "- The missing-group penalty cannot be checked on TAWOS (every historical story has all its features); it "
        "is checked with fault-injection tests of the service (NFR7).",
        "- The dashboard shows the band with its interval and probability; the user study (NFR4) checks that "
        "practitioners read it as intended.", "",
    ])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    arena = ArenaData.load()
    names = [c.id for c in configs.CONFIGS if (MODELS_DIR / c.id / predictor.MANIFEST).exists()]
    for name in names:
        fit(arena, name)
    results = [evaluate(arena, name) for name in names]
    board = MODELS_DIR / "leaderboard.json"
    winner = json.loads(board.read_text(encoding="utf-8"))["pooled_winner"] if board.exists() else None
    args.report.write_text(build_report(results, winner), encoding="utf-8")
    log(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
