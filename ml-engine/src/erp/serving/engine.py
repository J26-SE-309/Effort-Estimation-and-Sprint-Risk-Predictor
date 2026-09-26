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

from erp.features import catalog
from erp.serving import explain, live, recommend, sprint
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
                 = None, pinned: str | None = None) -> dict:
        features, sources = live.build(project_id, stories, team, sprint_context)
        history = features["history_sprints"].iloc[0] if len(features) else None
        choice = self.router.choose(project_id, None if pd.isna(history) else float(history), pinned)
        return self._predict(choice.config, choice.mode, choice.reason, project_id, features, sources)

    def compare(self, project_id: str, stories: list[dict], configs: list[str] | None = None,
                team: dict | None = None, sprint_context: dict | None = None) -> dict:
        """The same backlog through several configurations, side by side (FR12)."""
        features, sources = live.build(project_id, stories, team, sprint_context)
        chosen = configs or self.router.available
        unknown = [c for c in chosen if c not in self.router.available]
        if unknown:
            raise KeyError(f"not available: {', '.join(unknown)}")
        return {"results": [self._predict(c, "pinned", "requested for comparison", project_id, features, sources)
                            for c in chosen]}

    def sprint_risk(self, project_id: str, stories: list[dict], team: dict | None = None,
                    sprint_context: dict | None = None, pinned: str | None = None) -> dict:
        """The sprint-level risk of committing to this backlog (FR16), with the story predictions it rests on."""
        result = self.estimate(project_id, stories, team, sprint_context, pinned)
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
        missing = live.missing_groups(sources)
        out = model.predict(features, text, missing_groups=missing.to_numpy())
        _, risk_factors, how = explain.factors(model, features, text)
        risk_reasons = explain.reasons(risk_factors)
        manifest = model.manifest
        configuration = {"encoder": manifest["encoder"]["name"], "learner": manifest["config"]["learner"],
                         "formulation": "multi_task" if "joint" in manifest else "single_task"}
        commit = (manifest.get("code") or {}).get("commit", "")[:7]
        version = f"{manifest['arena']}/{config}" + (f"@{commit}" if commit else "")
        now = datetime.now(UTC).isoformat(timespec="seconds")
        predictions = []
        for i, story_id in enumerate(features.index):
            row = out.iloc[i]
            points = float(row["predicted_story_points"])
            story = {"predicted_story_points": points, "effort_category": recommend.effort_category(points),
                     "interval_low": float(row["interval_0.8_low"]), "interval_high": float(row["interval_0.8_high"]),
                     "sprint_risk_level": row["risk_level"]}
            used = sources.iloc[i]
            predictions.append({
                "story_id": str(story_id),
                "project_id": project_id,
                "predicted_story_points": round(points, 2),
                "effort_category": story["effort_category"],
                "prediction_interval": {"lower": round(story["interval_low"], 2),
                                        "upper": round(story["interval_high"], 2)},
                "interval_coverage": 0.8,
                "sprint_risk_level": row["risk_level"],
                "spillover_probability": round(float(row["spillover_probability"]), 4),
                "at_risk": bool(row["at_risk"]),
                "confidence_score": round(float(row["confidence_score"]), 4),
                "confidence_level": row["confidence"],
                "key_risk_reasons": [{"factor": r["factor"], "direction": "increases", "weight": r["share"]}
                                     for r in risk_reasons[i]],
                "recommendations": recommend.for_story(story, features.iloc[i], risk_reasons[i]),
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


def default_models_dir() -> Path:
    """The arena's models inside this repository (the service's MODELS_DIR setting can point elsewhere)."""
    from erp import config
    from erp.arena.configs import ARENA

    return config.MODEL_BUNDLES_DIR / ARENA
