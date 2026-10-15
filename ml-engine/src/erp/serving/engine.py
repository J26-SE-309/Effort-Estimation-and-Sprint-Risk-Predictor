"""The prediction engine behind the service: from a backlog to the proposal's Appendix C predictions.

For one request (one project's backlog): features for the live stories (live.py), the configuration the router
picks (router.py), predictions with C2 intervals, C1 probabilities and the confidence score (arena.predictor),
explanations as planning factors (explain.py), recommendations (recommend.py), and on demand the sprint-level
simulation (sprint.py). The engine knows nothing about HTTP; the FastAPI service in backend/ calls it.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from erp.arena.predictor import Predictor, finish
from erp.features import catalog
from erp.models import confidence as trust
from erp.serving import explain, live, recommend, sprint, whatif
from erp.serving.router import Router

GROUP_NAMES = {  # catalogue groups -> the names the API reports (FR5's feature groups, FR17)
    "text": "textual", "requirement_quality": "requirement_quality", "acceptance_criteria": "acceptance_criteria",
    "traceability": "traceability", "dependencies": "dependency", "team_history": "historical_sprint",
    "sprint": "sprint_context", "metadata": "metadata",
}


class Engine:
    def __init__(self, models_dir: Path, preload: bool = True):
        self.router = Router(models_dir)
        if preload:  # the pooled winner answers most requests; load it before the first one arrives
            self.router.load(self.router.pooled_winner)

    # ------------------------------------------------------------------ predictions

    def estimate(self, project_id: str, stories: list[dict], team: dict | None = None, sprint_context: dict | None
                 = None, pinned: str | None = None, focus: list[str] | None = None) -> dict:
        features, sources = _focused(*live.build(project_id, stories, team, sprint_context), focus)
        history = features["history_sprints"].iloc[0] if len(features) else None
        choice = self.router.choose(project_id, None if pd.isna(history) else float(history), pinned)
        return self._predict(choice.config, choice.mode, choice.reason, project_id, features, sources)

    def compare(self, project_id: str, stories: list[dict], configs: list[str] | None = None,
                team: dict | None = None, sprint_context: dict | None = None, focus: list[str] | None = None) -> dict:
        """The same backlog through several configurations, side by side (FR12)."""
        features, sources = _focused(*live.build(project_id, stories, team, sprint_context), focus)
        chosen = configs or self.router.available
        unknown = [c for c in chosen if c not in self.router.available]
        if unknown:
            raise KeyError(f"not available: {', '.join(unknown)}")
        return {"results": [self._predict(c, "pinned", "requested for comparison", project_id, features, sources)
                            for c in chosen]}

    def sprint_risk(self, project_id: str, stories: list[dict], team: dict | None = None,
                    sprint_context: dict | None = None, pinned: str | None = None,
                    focus: list[str] | None = None) -> dict:
        """The sprint-level risk of committing to this backlog (FR16), with the story predictions it rests on
        (with focus, only the focus stories are simulated)."""
        result = self.estimate(project_id, stories, team, sprint_context, pinned, focus)
        predictions = result["predictions"]
        capacity = (sprint_context or {}).get("capacity_points") or (team or {}).get("velocity_mean")
        simulation = sprint.simulate(
            np.log1p([p["predicted_story_points"] for p in predictions]),
            [p["prediction_interval"]["upper"] for p in predictions],
            [p["spillover_probability"] for p in predictions],
            capacity, (team or {}).get("velocity_variance"))
        return {**result, "sprint": {**simulation, "recommendations": recommend.sprint_level(simulation)}}

    # ------------------------------------------------------------------ inner workings

    def _predict(self, config: str, mode: str, reason: str, project_id: str, features: pd.DataFrame,
                 sources: pd.DataFrame) -> dict:
        model = self.router.load(config)
        text = explain.encode(model, features)
        missing = live.missing_groups(sources).to_numpy()
        run = None
        if isinstance(model, Predictor):  # one pass serves both the prediction and its explanation
            run = model.run(features, text)
            out = finish(model, features, run.log_points, run.raw, missing)
        else:
            out = model.predict(features, text, missing_groups=missing)
        _, risk_factors, how = explain.factors(model, features, text, run=run, tasks=("risk",))
        risk_reasons = explain.reasons(risk_factors)
        manifest = model.manifest
        configuration = {"encoder": manifest["encoder"]["name"], "learner": manifest["config"]["learner"],
                         "formulation": "multi_task" if "joint" in manifest else "single_task"}
        commit = (manifest.get("code") or {}).get("commit", "")[:7]
        version = f"{manifest['arena']}/{config}" + (f"@{commit}" if commit else "")
        now = datetime.now(UTC).isoformat(timespec="seconds")
        stories, advice = [], []
        for i in range(len(features)):
            row = out.iloc[i]
            points = float(row["predicted_story_points"])
            stories.append({"predicted_story_points": points, "effort_category": recommend.effort_category(points),
                            "interval_low": float(row["interval_0.8_low"]),
                            "interval_high": float(row["interval_0.8_high"]), "sprint_risk_level": row["risk_level"]})
            advice.append(recommend.for_story(stories[i], features.iloc[i], risk_reasons[i]))
        effects = _effects(model, features, text, missing, stories, advice)
        predictions = []
        for i, story_id in enumerate(features.index):
            row = out.iloc[i]
            story = stories[i]
            used = sources.iloc[i]
            predictions.append({
                "story_id": str(story_id),
                "project_id": project_id,
                "predicted_story_points": round(story["predicted_story_points"], 2),
                "effort_category": story["effort_category"],
                "prediction_interval": {"lower": round(story["interval_low"], 2),
                                        "upper": round(story["interval_high"], 2)},
                "interval_coverage": 0.8,
                "sprint_risk_level": row["risk_level"],
                "spillover_probability": round(float(row["spillover_probability"]), 4),
                "at_risk": bool(row["at_risk"]),
                "confidence_score": round(float(row["confidence_score"]), 4),
                "confidence_level": row["confidence"],
                "confidence_breakdown": _breakdown(model, row, features.iloc[i], int(missing[i])),
                "key_risk_reasons": [{"factor": r["factor"], "direction": "increases", "weight": r["share"]}
                                     for r in risk_reasons[i]],
                "recommendations": _by_effect(advice[i], effects, i, float(row["spillover_probability"])),
                "risk_if_all_done": effects.get((i, None)),
                "model_configuration": configuration,
                "configuration_id": config,
                "selection_mode": mode,
                "selection_reason": reason,
                "feature_groups_used": [GROUP_NAMES[g] for g, s in used.items() if s not in ("proxy", "missing")],
                "degraded_feature_groups": [GROUP_NAMES[g] for g, s in used.items() if s in ("proxy", "missing")],
                "feature_sources": {GROUP_NAMES[g]: s for g, s in used.items()},
                "explanation_method": how,
                "model_version": version,
                "generated_at": now,
            })
        # the features the models saw, per story, for the service's audit log (FR21); not part of the response
        snapshot = json.loads(features[catalog.names()].to_json(orient="index"))
        return {"configuration_id": config, "selection_mode": mode, "selection_reason": reason,
                "predictions": predictions, "features": snapshot}

    # ------------------------------------------------------------------ the arena, for /models

    def models(self) -> dict:
        board = self.router.leaderboard
        summaries = []
        for config, metrics in board["configs"].items():
            summaries.append({"configuration_id": config, "label": metrics["label"], "role": metrics["role"],
                              "encoder": metrics["encoder"], "learner": metrics["learner"],
                              "status": "available" if config in self.router.available else "not installed",
                              "loaded": config in self.router.loaded(),
                              "eligible": metrics["eligible"], "failed_requirements": metrics["failed_requirements"],
                              "composite": metrics["composite"],
                              "metrics": {k: metrics[k] for k in ("mae", "sa", "roc_auc", "f1", "ece", "coverage_0.8",
                                                                  "latency_p95") if k in metrics}})
        return {"arena": board["arena"], "pooled_winner": self.router.pooled_winner, "weights": board["weights"],
                "configurations": sorted(summaries, key=lambda s: -s["composite"])}


def _effects(model, features: pd.DataFrame, text, missing, stories: list[dict], advice: list[list[dict]]) -> dict:
    """{(story position, action or None for all of them): the risk expected once done} (erp.serving.whatif): the
    changed stories go through the configuration in one more pass."""
    changed, origins, keys = whatif.scenarios(features, advice, stories)
    if not keys:
        return {}
    out = model.predict(changed, whatif.take(text, origins, changed.index),
                        missing_groups=np.asarray(missing)[origins])
    return {key: {"spillover_probability": round(float(out["spillover_probability"].iloc[n]), 4),
                  "sprint_risk_level": out["risk_level"].iloc[n]} for n, key in enumerate(keys)}


def _by_effect(advice: list[dict], effects: dict, i: int, probability: float) -> list[dict]:
    """The story's recommendations with their expected effect, the one expected to lower the risk most first."""
    out = [{**rec, "expected_effect": effects.get((i, rec["action"]))} for rec in advice]
    return sorted(out, key=lambda rec: (rec["expected_effect"] or {}).get("spillover_probability", probability))


def _breakdown(model, row: pd.Series, story: pd.Series, missing: int) -> dict | None:
    """The confidence score's parts (erp.models.confidence): S x (E + R) / 2, with what lowered S."""
    if getattr(model, "confidence", None) is None or "data_support" not in row:
        return None
    penalties = []
    sprints = story.get("history_sprints")
    sprints = 0 if sprints is None or pd.isna(sprints) else int(sprints)
    if sprints < trust.COLD_START_SPRINTS:
        penalties.append({"reason": "cold_start", "factor": trust.PENALTIES["cold_start"],
                          "detail": f"the team has {sprints} closed sprint(s); {trust.COLD_START_SPRINTS} are needed"})
    if str(story.get("project_key")) not in model.confidence.known_projects:
        penalties.append({"reason": "unseen_project", "factor": trust.PENALTIES["unseen_project"],
                          "detail": "the models were trained on other projects' stories"})
    if missing:
        penalties.append({"reason": "missing_group", "factor": round(trust.PENALTIES["missing_group"] ** missing, 4),
                          "detail": f"{missing} feature group(s) not available"})
    return {"effort_certainty": round(float(row["effort_certainty"]), 4),
            "risk_certainty": round(float(row["risk_certainty"]), 4),
            "data_support": round(float(row["data_support"]), 4), "penalties": penalties}


def _focused(features: pd.DataFrame, sources: pd.DataFrame, focus: list[str] | None
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only the focus stories' rows. Their features were computed with the whole backlog, so the rest of the
    sprint still counts as committed work (sprint_committed_points) without being predicted itself."""
    if not focus:
        return features, sources
    unknown = [story_id for story_id in focus if story_id not in features.index]
    if unknown:
        raise KeyError(f"focus stories not in the request: {', '.join(unknown)}")
    keep = list(dict.fromkeys(focus))
    return features.loc[keep], sources.loc[keep]


def default_models_dir() -> Path:
    """The arena's models inside this repository (the service's MODELS_DIR setting can point elsewhere)."""
    from erp import config
    from erp.arena.configs import ARENA

    return config.MODEL_BUNDLES_DIR / ARENA
