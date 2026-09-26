"""Phase 4: the hypothesis experiments H1 and H2 (ML guide 8.2).

H1 - does learning effort and risk together help? M3, the joint network, against the same network trained for
     one task at a time: identical inputs, architecture, splits and seeds (ML guide 4.3). The single-task twins
     get the same tuning budget as M3 on the same inner folds. Each of the three is then trained with seeds 1-10
     and the ten paired results are compared with the Wilcoxon signed-rank test and A12. Tree learners cannot
     share a representation between tasks, so H1 is tested only inside the neural-network family.
H2 - do the requirement-quality and traceability signals add information? The best single configuration per
     task (lowest test MAE for effort, highest test ROC-AUC for risk; the stack, M3 and DistilBERT have no single
     feature list and are left out) is retrained without each upstream feature group and without all three,
     with its tuned settings. Five seeds per variant are averaged, so seed noise cannot hide or fake a
     difference; differences against the full model get 95% paired bootstrap intervals over the test stories.
     How much the full model leans on each group: group permutation importance on the test split (the
     group's columns shuffled together across stories; works for every learner), plus the group's share of
     the SHAP values when the learner computes them natively (the boosters).

Usage:
    erp-run-hypotheses              # after erp-arena-report; writes ml-engine/reports/hypotheses.md
    erp-run-hypotheses --only h1    # or h2
    erp-run-hypotheses --report-only  # rebuild the report from the saved results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from erp import config as paths
from erp.arena import configs, train
from erp.arena.data import MODELS_DIR, WORK_DIR, ArenaData, targets, text_columns
from erp.explore.profile_tawos import md_table
from erp.features import catalog
from erp.models import calibration, metrics
from erp.models.inputs import design
from erp.models.learners import LEARNERS
from erp.models.mlp import MLP

H1_SEEDS = list(range(1, 11))
H2_SEEDS = list(range(1, 6))
H2_VARIANTS = {
    "all features": (),
    "without requirement quality (Component 1)": tuple(catalog.names("requirement_quality")),
    "without acceptance criteria (Component 2)": tuple(catalog.names("acceptance_criteria")),
    "without traceability (Component 3)": tuple(catalog.names("traceability")),
    "without all three upstream groups": tuple(n for g in catalog.UPSTREAM_GROUPS for n in catalog.names(g)),
}
OUT_DIR = WORK_DIR / "hypotheses"
REPORT = paths.REPO_ROOT / "ml-engine" / "reports" / "hypotheses.md"


def risk_scores(arena: ArenaData, raw: np.ndarray) -> dict:
    """C1 and the threshold on the calibration split, then the test metrics."""
    cal, test = arena.mask("cal"), arena.mask("test")
    y = targets(arena.frame, "risk")
    calibrator = calibration.Calibrator("auto").fit(raw[cal], y[cal])
    probability = calibrator.predict(raw)
    threshold = metrics.choose_threshold(y[cal], probability[cal])
    return {**metrics.classification_report(y[test], probability[test], threshold), "probability": probability}


def effort_scores(arena: ArenaData, log_points: np.ndarray) -> dict:
    test = arena.frame[arena.mask("test")]
    return metrics.effort_report(test["story_points"], np.expm1(log_points[arena.mask("test")]), test["Project_ID"])


# ---------------------------------------------------------------- H1


def h1(arena: ArenaData, trials: int, retune: bool) -> dict:
    m3 = json.loads((MODELS_DIR / "sbert-mtl" / "model.json").read_text(encoding="utf-8"))
    tuned_path = OUT_DIR / "h1-tuning.json"
    tuned = json.loads(tuned_path.read_text(encoding="utf-8")) if tuned_path.exists() and not retune else {}
    config = configs.BY_ID["sbert-mtl"]
    for task in ("effort", "risk"):
        if task not in tuned:
            study, _, summary = train.tune_joint(config, arena, trials, tasks=(task,))
            tuned[task] = summary
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            tuned_path.write_text(json.dumps(tuned, indent=1), encoding="utf-8")
    variants = {"joint (M3)": (("effort", "risk"), m3["tuning"]["best_params"]),
                "effort only": (("effort",), tuned["effort"]["best_params"]),
                "risk only": (("risk",), tuned["risk"]["best_params"])}

    x = design(arena.frame, arena.full_text("sbert"), "joint", text_columns=text_columns("sbert"),
               levels=arena.levels)
    tr, cal = arena.mask("train"), arena.mask("cal")
    y = {task: targets(arena.frame, task) for task in ("effort", "risk")}
    rows = []
    for seed in H1_SEEDS:
        for name, (tasks, params) in variants.items():
            model = MLP(tasks).fit(x[tr], {t: y[t][tr] for t in tasks}, x[cal], {t: y[t][cal] for t in tasks},
                                   params, seed=seed)
            out = model.predict(x)
            row = {"seed": seed, "variant": name, "epochs": model.best_epoch}
            if "effort" in tasks:
                e = effort_scores(arena, out["effort"])
                row.update({"mae": e["mae"], "sa": e["sa"]})
            if "risk" in tasks:
                r = risk_scores(arena, out["risk"])
                row.update({k: r[k] for k in ("f1", "roc_auc", "pr_auc", "ece")})
            rows.append(row)
        train.log(f"H1 seed {seed} done")
    per_seed = pd.DataFrame(rows)
    comparisons = []
    for metric, single, better in (("mae", "effort only", "lower"), ("sa", "effort only", "higher"),
                                   ("f1", "risk only", "higher"), ("roc_auc", "risk only", "higher"),
                                   ("pr_auc", "risk only", "higher"), ("ece", "risk only", "lower")):
        joint = per_seed[per_seed["variant"] == "joint (M3)"].set_index("seed")[metric]
        alone = per_seed[per_seed["variant"] == single].set_index("seed")[metric]
        diff = joint - alone
        p = float("nan") if np.allclose(diff, 0) else metrics.wilcoxon_p(joint, alone)
        comparisons.append({"metric": metric, "better": better, "joint_mean": joint.mean(), "joint_sd": joint.std(),
                            "single": single, "single_mean": alone.mean(), "single_sd": alone.std(),
                            "difference": diff.mean(), "p": p, "a12": metrics.vargha_delaney(joint, alone),
                            "joint_wins": int(((diff < 0) if better == "lower" else (diff > 0)).sum())})
    result = {"tuning": {"joint (M3)": m3["tuning"], **{f"{t} only": tuned[t] for t in ("effort", "risk")}},
              "per_seed": per_seed.to_dict("records"), "comparisons": comparisons}
    (OUT_DIR / "h1.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    return result


# ---------------------------------------------------------------- H2


def group_shares(learner, x: pd.DataFrame) -> pd.Series:
    """Share of the mean absolute SHAP value per feature group (the text vector counts as one group)."""
    group_of = {f.name: f.group for f in catalog.FEATURES}
    shap = learner.contributions(x).abs().mean()
    groups = shap.groupby(lambda c: group_of.get(c.split("=", 1)[0], "text vector (encoder)")).sum()
    return (groups / groups.sum()).sort_values(ascending=False)


PERMUTATION_REPEATS = 5


def feature_groups(columns) -> dict[str, list[str]]:
    group_of = {f.name: f.group for f in catalog.FEATURES}
    groups: dict[str, list[str]] = {}
    for column in columns:
        groups.setdefault(group_of.get(column.split("=", 1)[0], "text vector (encoder)"), []).append(column)
    return groups


def group_permutation(learner, x: pd.DataFrame, task: str, arena: ArenaData) -> dict:
    """Loss in test accuracy when one feature group is shuffled across stories (all its columns together)."""
    from sklearn.metrics import roc_auc_score

    test = arena.frame.loc[x.index]

    def score(predicted):
        if task == "effort":
            return -metrics.mae(test["story_points"], np.expm1(predicted))  # higher is better
        return roc_auc_score(test["at_risk"], predicted)

    base = score(learner.predict(x))
    rng = np.random.default_rng(configs.SEED)
    result = {}
    for group, columns in feature_groups(x.columns).items():
        losses = []
        for _ in range(PERMUTATION_REPEATS):
            order = rng.permutation(len(x))
            shuffled = x.copy()
            for column in columns:
                shuffled[column] = x[column].iloc[order].set_axis(x.index)
            losses.append(base - score(learner.predict(shuffled)))
        result[group] = {"mean": float(np.mean(losses)), "sd": float(np.std(losses))}
    return dict(sorted(result.items(), key=lambda kv: -kv[1]["mean"]))


def h2(arena: ArenaData) -> dict:
    board = json.loads((MODELS_DIR / "leaderboard.json").read_text(encoding="utf-8"))["configs"]
    single = {n: c for n, c in board.items() if c["learner"] in LEARNERS}
    best = {"effort": min(single, key=lambda n: single[n]["mae"]),
            "risk": max(single, key=lambda n: single[n]["roc_auc"])}
    tr, cal, test = arena.mask("train"), arena.mask("cal"), arena.mask("test")
    y_test = targets(arena.frame, "risk")[test]
    points = arena.frame["story_points"].to_numpy()[test]
    result = {"best": best, "tasks": {}}
    for task, name in best.items():
        cfg = configs.BY_ID[name]
        manifest = json.loads((MODELS_DIR / name / "model.json").read_text(encoding="utf-8"))
        params = manifest[task]["tuning"]["best_params"]
        text = arena.full_text(cfg.encoder)
        y = targets(arena.frame, task)
        seeds = [configs.SEED] if cfg.learner == "svm" else H2_SEEDS  # the SVM has no randomness
        variants, shares, permutation = {}, None, None
        for variant, drop in H2_VARIANTS.items():
            x = design(arena.frame, text, task, text_columns=text_columns(cfg.encoder), levels=arena.levels,
                       drop=drop)
            runs, per_seed = [], []
            for seed in seeds:
                learner = LEARNERS[cfg.learner](task).fit(x[tr], y[tr], x[cal], y[cal], params, seed=seed)
                predicted = learner.predict(x)
                runs.append(predicted)
                per_seed.append(effort_scores(arena, predicted)["mae"] if task == "effort"
                                else risk_scores(arena, predicted)["roc_auc"])
                if variant == "all features" and seed == seeds[0]:
                    permutation = group_permutation(learner, x[test], task, arena)
                    if hasattr(learner, "contributions"):
                        shares = group_shares(learner, x[test])
            mean = np.mean(runs, axis=0)
            scored = effort_scores(arena, mean) if task == "effort" else risk_scores(arena, mean)
            variants[variant] = {"prediction": mean, "scores": scored, "seed_sd": float(np.std(per_seed))}
            train.log(f"H2 {task} {variant} done")
        full = variants["all features"]
        rows = []
        for variant, v in variants.items():
            row = {"variant": variant, "seed_sd": v["seed_sd"]}
            if task == "effort":
                row["mae"] = v["scores"]["mae"]
                if variant != "all features":
                    row["mae_change"] = metrics.paired_bootstrap(
                        metrics.mae, points, np.expm1(v["prediction"][test]), np.expm1(full["prediction"][test]))
            else:
                row.update({k: v["scores"][k] for k in ("roc_auc", "pr_auc", "f1", "ece")})
                if variant != "all features":
                    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

                    a, b = v["scores"]["probability"][test], full["scores"]["probability"][test]
                    row["roc_auc_change"] = metrics.paired_bootstrap(roc_auc_score, y_test, a, b)
                    row["pr_auc_change"] = metrics.paired_bootstrap(average_precision_score, y_test, a, b)
                    flags_a = (a >= v["scores"]["threshold"]).astype(float)
                    flags_b = (b >= full["scores"]["threshold"]).astype(float)
                    row["f1_change"] = metrics.paired_bootstrap(lambda t, s: f1_score(t, s >= 0.5), y_test,
                                                                flags_a, flags_b)
            rows.append(row)
        result["tasks"][task] = {"config": name, "seeds": seeds, "variants": rows, "permutation": permutation,
                                 "shap_group_shares": None if shares is None else shares.round(4).to_dict()}
    prevalence = {f.name: float((arena.frame[f.name].astype(float) > 0).mean())
                  for g in catalog.UPSTREAM_GROUPS for f in catalog.FEATURES if f.group == g}
    result["prevalence"] = prevalence
    (OUT_DIR / "h2.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    return result


# ---------------------------------------------------------------- report


def change(c: dict, digits: int = 3) -> str:
    return f"{c['difference']:+.{digits}f} ({c['low']:+.{digits}f} to {c['high']:+.{digits}f}), p = {c['p']:.2g}"


def build_report(one: dict | None, two: dict | None) -> str:
    labels = {c.id: c.label for c in configs.CONFIGS}
    lines = ["# Hypotheses H1 and H2 (Phase 4)", "",
             "Generated by `erp-run-hypotheses` after the arena (`arena.md`). Test split only, as in the arena. "
             "H3 (are the explanations meaningful?) needs the practitioner study of Phase 6; the evidence for H4 "
             "(is the uncertainty trustworthy?) is the coverage and calibration in `arena.md` and "
             "`uncertainty-and-explanations.md`.", ""]
    if one:
        names = {"mae": "MAE (points)", "sa": "SA", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
                 "ece": "ECE"}
        rows = [{"Metric": names[c["metric"]], "Better": c["better"],
                 "Joint (M3)": f"{c['joint_mean']:.3f} ± {c['joint_sd']:.3f}",
                 "Single-task": f"{c['single_mean']:.3f} ± {c['single_sd']:.3f}",
                 "Joint better in": f"{c['joint_wins']} of {len(H1_SEEDS)} seeds",
                 "Wilcoxon p": "–" if np.isnan(c["p"]) else f"{c['p']:.2g}", "A12": f"{c['a12']:.2f}"}
                for c in one["comparisons"]]
        tuning = one["tuning"]
        lines += [
            "## H1: does learning effort and risk together help?", "",
            "The same network (SBERT + structured features → 256 → 128, one head per task) trained jointly (M3) and "
            "once per task, on the same inputs (no story points for any of them, since they are the effort "
            "head's answer), splits and seeds. Each got its own Optuna tuning with the same budget "
            f"({configs.N_TRIALS} trials on the same inner folds); then each was trained with seeds "
            f"{H1_SEEDS[0]}–{H1_SEEDS[-1]}. Mean ± standard deviation over the ten seeds; the Wilcoxon signed-rank "
            "test pairs the runs by seed. A12 is the chance that a joint run scores higher than a single-task run.",
            "",
            md_table(pd.DataFrame(rows)), "",
            "Tuned settings: " + "; ".join(
                f"{name}: " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                                        for k, v in t["best_params"].items()) for name, t in tuning.items()) + ".",
            "",
            "Tree learners cannot share a learned representation between the two tasks, so H1 is answered inside "
            "the neural-network family only, as the ML guide (4.3) recommends; the arena shows how the networks "
            "compare with the tree learners.", ""]
    if two:
        lines += ["## H2: do the requirement-quality and traceability signals add information?", "",
                  "Ablation of the upstream feature groups (proxies for Components 1–3) on the best single "
                  "configuration per task, retrained with its tuned settings. Predictions are the mean of "
                  f"{len(H2_SEEDS)} seeds (1 for the SVM, which has no randomness); 'seed sd' is the spread of the "
                  "single-seed results, the size of difference that seed luck alone can produce. Changes are "
                  "variant minus full model with a 95% paired bootstrap interval and p-value.", ""]
        for task, t in two["tasks"].items():
            label = labels[t["config"]]
            rows = []
            for v in t["variants"]:
                if task == "effort":
                    rows.append({"Variant": v["variant"], "MAE": f"{v['mae']:.3f}", "Seed sd": f"{v['seed_sd']:.3f}",
                                 "MAE change": change(v["mae_change"]) if "mae_change" in v else "–"})
                else:
                    rows.append({"Variant": v["variant"], "ROC-AUC": f"{v['roc_auc']:.3f}",
                                 "Seed sd": f"{v['seed_sd']:.3f}", "F1": f"{v['f1']:.3f}",
                                 "ROC-AUC change": change(v["roc_auc_change"]) if "roc_auc_change" in v else "–",
                                 "F1 change": change(v["f1_change"]) if "f1_change" in v else "–"})
            lines += [f"### {'Effort (M1)' if task == 'effort' else 'Risk (M2)'}: {label}", "",
                      md_table(pd.DataFrame(rows)), ""]
            if t.get("permutation"):
                unit = "MAE increase (points)" if task == "effort" else "ROC-AUC loss"
                lines += [f"How much the full model leans on each group: {unit} when the group is shuffled across "
                          f"the test stories (mean ± sd over {PERMUTATION_REPEATS} shuffles).", "",
                          md_table(pd.DataFrame({"Group": list(t["permutation"]), unit: [
                              f"{v['mean']:+.4f} ± {v['sd']:.4f}" for v in t["permutation"].values()]})), ""]
            if t["shap_group_shares"]:
                shares = ", ".join(f"{g} {s:.1%}" for g, s in t["shap_group_shares"].items())
                lines += [f"Share of the full model's mean absolute SHAP value by feature group: {shares}.", ""]
        prevalence = two["prevalence"]
        lines += ["How often the upstream proxies are non-zero at all (all stories):", "",
                  md_table(pd.DataFrame({"Feature": list(prevalence), "Non-zero": [f"{v:.1%}" for v in
                                                                                   prevalence.values()]})), "",
                  "A feature that is almost always zero cannot carry much signal: only "
                  f"{prevalence['has_acceptance_criteria']:.1%} of the stories have acceptance criteria, so the "
                  "Component 2 ablation says little about acceptance criteria "
                  "themselves (the open question for the supervisor). A negative or null result here is a finding "
                  "about proxies computed from open-source issue text, not about Components 1–3; it is rerun on "
                  "their real batch scores when they are available (ML guide 6.6).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["h1", "h2"])
    parser.add_argument("--trials", type=int, default=configs.N_TRIALS)
    parser.add_argument("--retune", action="store_true", help="tune the H1 single-task networks again")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from the saved results")
    args = parser.parse_args(argv)

    def load(name):
        path = OUT_DIR / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    if args.report_only:
        one, two = load("h1"), load("h2")
    else:
        arena = ArenaData.load()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        one = h1(arena, args.trials, args.retune) if args.only in (None, "h1") else load("h1")
        two = h2(arena) if args.only in (None, "h2") else load("h2")
    args.report.write_text(build_report(one, two), encoding="utf-8")
    train.log(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
