"""Version 1 of the prediction API (proposal Figure 2: /estimate, /risk, /recommend, /models, /compare).

Until the trained models are plugged in, /estimate returns placeholder predictions
(model_version "stub") so the gateway and dashboard can be built against the real contract.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    EstimateRequest,
    EstimateResponse,
    ModelConfiguration,
    ModelSummary,
    Prediction,
    PredictionInterval,
    StoryInput,
)

router = APIRouter(tags=["effort estimation and sprint risk"])

# Proposal Appendix D: the configurations the Comparative Model Arena starts with.
ARENA_CONFIGURATIONS = [
    ("tfidf", "random_forest", "single_task"),
    ("tfidf", "svr_svm", "single_task"),
    ("fasttext", "lightgbm", "single_task"),
    ("sbert", "xgboost", "single_task"),
    ("sbert", "lightgbm", "single_task"),
    ("sbert", "catboost", "single_task"),
    ("sbert", "mlp", "multi_task"),
    ("distilbert", "fine_tuned", "single_task"),
    ("stacked", "ensemble", "single_task"),
]

_QUALITY_FIELDS = ("ambiguity_score", "vague_term_count", "missing_info_flag_count", "ac_completeness_score",
                   "invest_compliance_flags")
_TRACEABILITY_FIELDS = ("traceability_coverage_pct", "unlinked_artifact_count", "has_linked_tests")


def feature_groups_used(story: StoryInput) -> list[str]:
    """Feature groups available for this story; missing upstream groups mean a degraded prediction (FR17)."""
    groups = ["textual", "dependency", "metadata"]
    if any(getattr(story.upstream, name) is not None for name in _QUALITY_FIELDS):
        groups.append("requirement_quality")
    if any(getattr(story.upstream, name) is not None for name in _TRACEABILITY_FIELDS):
        groups.append("traceability")
    return groups


def _placeholder(project_id: str, story: StoryInput, now: datetime) -> Prediction:
    return Prediction(
        story_id=story.story_id,
        project_id=project_id,
        predicted_story_points=3.0,
        effort_category="medium",
        prediction_interval=PredictionInterval(lower=1.0, upper=5.0),
        sprint_risk_level="low",
        spillover_probability=0.0,
        confidence_score=0.0,
        key_risk_reasons=[],
        recommendations=[],
        model_configuration=ModelConfiguration(encoder="none", learner="none", formulation="single_task"),
        selection_mode="auto",
        feature_groups_used=feature_groups_used(story),
        model_version="stub",
        generated_at=now,
    )


@router.post("/estimate", response_model=EstimateResponse)
def estimate(request: EstimateRequest) -> EstimateResponse:
    """Estimate effort and sprint risk for every story in a backlog, in the order given."""
    now = datetime.now(UTC)
    return EstimateResponse(predictions=[_placeholder(request.project_id, story, now) for story in request.stories])


@router.get("/models", response_model=list[ModelSummary])
def models() -> list[ModelSummary]:
    """Configurations in the Comparative Model Arena. The leaderboard metrics arrive with the arena."""
    return [
        ModelSummary(
            configuration=ModelConfiguration(encoder=encoder, learner=learner, formulation=formulation),
            status="planned",
        )
        for encoder, learner, formulation in ARENA_CONFIGURATIONS
    ]


def _not_implemented() -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "Not implemented yet")


@router.post("/risk")
def sprint_risk() -> None:
    """Sprint-level risk for a whole backlog (FR16, Monte Carlo aggregator)."""
    _not_implemented()


@router.post("/recommend")
def recommend() -> None:
    """Planning recommendations for a backlog (FR15)."""
    _not_implemented()


@router.post("/compare")
def compare() -> None:
    """Predictions from several configurations side by side (FR12)."""
    _not_implemented()
