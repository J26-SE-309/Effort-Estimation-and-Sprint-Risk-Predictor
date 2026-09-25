"""Request and response models: the API contract of this service.

The prediction mirrors Appendix C of the proposal and the JSON Schemas in
Synapse-Web/contracts/effort-estimation; change both together.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UpstreamSignals(BaseModel):
    """Features supplied by Components 1-3 through the gateway. Any of them may be missing (FR17)."""

    ambiguity_score: float | None = Field(default=None, ge=0, le=1)
    vague_term_count: int | None = Field(default=None, ge=0)
    missing_info_flag_count: int | None = Field(default=None, ge=0)
    ac_completeness_score: float | None = Field(default=None, ge=0, le=1)
    invest_compliance_flags: dict[str, bool] | None = None
    traceability_coverage_pct: float | None = Field(default=None, ge=0, le=1)
    unlinked_artifact_count: int | None = Field(default=None, ge=0)
    has_linked_tests: bool | None = None


class StoryInput(BaseModel):
    story_id: str
    title: str = Field(min_length=1)
    description: str = ""
    acceptance_criteria: list[str] = []
    issue_type: str | None = None
    priority: str | None = None
    story_points: float | None = Field(default=None, ge=0, description="The team's estimate, if the story has one")
    blocker_count: int = Field(default=0, ge=0)
    dep_in_degree: int = Field(default=0, ge=0)
    dep_out_degree: int = Field(default=0, ge=0)
    upstream: UpstreamSignals = UpstreamSignals()


class EstimateRequest(BaseModel):
    project_id: str
    sprint_id: str | None = None
    stories: list[StoryInput] = Field(min_length=1, max_length=200)


class PredictionInterval(BaseModel):
    lower: float = Field(ge=0)
    upper: float = Field(ge=0)


class RiskReason(BaseModel):
    factor: str = Field(description="Planning factor shown to users, e.g. 'Requirement ambiguity'")
    direction: Literal["increases", "decreases"]
    weight: float = Field(ge=0, le=1)


RecommendationAction = Literal[
    "split_story",
    "refine_requirement",
    "add_acceptance_criteria",
    "create_tests",
    "resolve_dependency",
    "reduce_sprint_scope",
]


class Recommendation(BaseModel):
    action: RecommendationAction
    message: str
    triggered_by: str


class ModelConfiguration(BaseModel):
    encoder: str
    learner: str
    formulation: Literal["single_task", "multi_task", "chained"]


class Prediction(BaseModel):
    story_id: str
    project_id: str
    predicted_story_points: float = Field(ge=0)
    effort_category: Literal["small", "medium", "large", "extra_large"]
    prediction_interval: PredictionInterval
    sprint_risk_level: Literal["low", "medium", "high"]
    spillover_probability: float = Field(ge=0, le=1)
    confidence_score: float = Field(ge=0, le=1)
    key_risk_reasons: list[RiskReason]
    recommendations: list[Recommendation]
    model_configuration: ModelConfiguration
    selection_mode: Literal["auto", "pinned"]
    feature_groups_used: list[str]
    model_version: str
    generated_at: datetime


class EstimateResponse(BaseModel):
    predictions: list[Prediction]


class ModelSummary(BaseModel):
    configuration: ModelConfiguration
    status: Literal["available", "planned"]
