"""Request and response models: the API contract of this service.

The prediction mirrors Appendix C of the proposal and the JSON Schemas in
Synapse-Web/contracts/effort-estimation; change both together. Fields added after the scaffold are optional
on requests and additive on responses, so existing callers (the gateway) keep working.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------------------------- requests


class UpstreamSignals(BaseModel):
    """Features supplied by Components 1-3 through the gateway. Any of them may be missing (FR17): the service
    then uses its own proxies computed from the story text and flags the group as degraded."""

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
    issue_type: str | None = Field(default=None, description="Story, Task, Bug, Improvement or New Feature")
    priority: str | None = Field(default=None, description="Jira priority name, e.g. Major or High")
    story_points: float | None = Field(default=None, ge=0, description="The team's estimate, if the story has one")
    blocker_count: int = Field(default=0, ge=0, description="Open issues this story is blocked by")
    dep_in_degree: int = Field(default=0, ge=0, description="Issues that depend on this story")
    dep_out_degree: int = Field(default=0, ge=0, description="Issues this story depends on")
    linked_issue_count: int | None = Field(default=None, ge=0, description="All linked issues, any link type")
    has_epic: bool | None = Field(default=None, description="The story belongs to an epic (unknown if left out)")
    in_progress: bool = Field(default=False, description="Work on the story has already started")
    added_mid_sprint: bool = Field(default=False, description="Added after the sprint started")
    days_into_sprint: float = Field(default=0, ge=0, description="Days after the sprint start it was added")
    upstream: UpstreamSignals = UpstreamSignals()


class TeamContext(BaseModel):
    """The team's recent delivery, as the platform knows it. Unknown history means a cold start: the pooled
    configuration answers and the confidence is lowered."""

    velocity_mean: float | None = Field(default=None, ge=0, description="Mean points finished, last 3 sprints")
    velocity_variance: float | None = Field(default=None, ge=0, description="Variance of points, last 5 sprints")
    closed_sprints: int | None = Field(default=None, ge=0, description="How many sprints the team has closed")
    mean_cycle_time_hours: float | None = Field(default=None, ge=0)
    spillover_rate: float | None = Field(default=None, ge=0, le=1, description="Share of recent stories spilled")
    reopen_rate: float | None = Field(default=None, ge=0, le=1)


class SprintContext(BaseModel):
    length_days: float | None = Field(default=None, gt=0)
    parallel_sprints: int | None = Field(default=None, ge=0, description="Other sprints of the project running")
    wip: int | None = Field(default=None, ge=0, description="Issues of the sprint already in progress")
    capacity_points: float | None = Field(default=None, gt=0, description="Planned capacity (default: velocity)")


class EstimateRequest(BaseModel):
    project_id: str
    sprint_id: str | None = None
    stories: list[StoryInput] = Field(min_length=1, max_length=200)
    team_context: TeamContext | None = None
    sprint_context: SprintContext | None = None
    pinned_configuration: str | None = Field(default=None, description="Answer with this configuration (FR12)")


class CompareRequest(EstimateRequest):
    configurations: list[str] | None = Field(default=None, description="Configuration ids; all available if empty")


# ---------------------------------------------------------------------------------------------- responses


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
    # Added in Phase 5 (all additive)
    prediction_id: str | None = Field(default=None, description="Audit record id (FR21); for feedback (FR19)")
    interval_coverage: float = Field(default=0.8, description="Share of stories whose points fall in the interval")
    at_risk: bool = Field(default=False, description="The risk passes the threshold chosen for recall >= 0.70")
    confidence_level: Literal["high", "medium", "low"] = Field(default="low", description="Band shown to users")
    configuration_id: str = ""
    selection_reason: str = ""
    degraded_feature_groups: list[str] = Field(default=[], description="Groups served by a proxy or missing (FR17)")
    feature_sources: dict[str, str] = Field(default={}, description="component, request, history, text, proxy or "
                                                                   "missing")
    explanation_method: str = ""


class EstimateResponse(BaseModel):
    predictions: list[Prediction]
    configuration_id: str = ""
    selection_mode: Literal["auto", "pinned"] = "auto"
    selection_reason: str = ""


class Quantiles(BaseModel):
    p10: float
    p50: float
    p90: float


class SprintRisk(BaseModel):
    """A1: does the whole commitment fit the team's capacity (FR16)? Monte Carlo over the stories' intervals."""

    stories: int
    runs: int
    committed_points: Quantiles
    capacity_points: Quantiles | None = Field(description="None without the team's velocity or a planned capacity")
    overcommit_probability: float | None = Field(ge=0, le=1)
    sprint_risk_level: Literal["low", "medium", "high"] | None
    expected_overflow_points: float | None
    overflow_if_over_p50: float | None
    expected_stories_at_risk: float
    stories_at_risk_p90: int
    recommendations: list[Recommendation]


class SprintRiskResponse(EstimateResponse):
    sprint: SprintRisk


class StoryRecommendations(BaseModel):
    story_id: str
    sprint_risk_level: Literal["low", "medium", "high"]
    key_risk_reasons: list[RiskReason]
    recommendations: list[Recommendation]


class RecommendResponse(BaseModel):
    stories: list[StoryRecommendations]
    sprint: list[Recommendation]


class CompareResult(BaseModel):
    configuration_id: str
    predictions: list[Prediction]


class CompareResponse(BaseModel):
    results: list[CompareResult]


class ModelSummary(BaseModel):
    configuration_id: str
    label: str
    role: str
    configuration: ModelConfiguration
    status: Literal["available", "not installed", "planned"]
    loaded: bool
    eligible: bool | None = None
    failed_requirements: list[str] = []
    composite: float | None = None
    metrics: dict[str, float] = {}


class ModelsResponse(BaseModel):
    arena: str
    pooled_winner: str
    weights: dict[str, float]
    configurations: list[ModelSummary]


class PinRequest(BaseModel):
    configuration_id: str


class Pin(BaseModel):
    project_id: str
    configuration_id: str | None
    pinned_at: datetime | None = None


class Feedback(BaseModel):
    """A product owner's decision on a prediction or one of its recommendations (FR19)."""

    prediction_id: str
    decision: Literal["accept", "adjust", "reject"]
    target: Literal["estimate", "risk", "recommendation"] = "estimate"
    recommendation_action: RecommendationAction | None = None
    adjusted_story_points: float | None = Field(default=None, ge=0)
    reason: str | None = None


class Outcome(BaseModel):
    """What really happened to the story in its sprint (FR19, for retraining, FR20)."""

    prediction_id: str
    completed_in_sprint: bool
    actual_story_points: float | None = Field(default=None, ge=0)
    reopened: bool = False


class Recorded(BaseModel):
    id: str
    recorded: bool = Field(description="False when the database is unavailable; nothing was stored")


# ---------------------------------------------------------------------------------------------- sprint history


class HistorySprint(BaseModel):
    sprint_id: str
    name: str | None = None
    started_at: datetime
    planned_end: datetime
    closed_at: datetime | None = Field(default=None, description="Empty while the sprint runs")
    stories: int
    committed_points: float = Field(description="Points of its stories when committed")
    completed_points: float | None = Field(default=None, description="Points finished in it, once closed (velocity)")
    spilled_over: int | None = Field(default=None, description="Stories not done by its end, once closed (R1)")


class HistorySummary(BaseModel):
    """What the models see about a project's team now (FR5), and its sprints, newest first."""

    project_id: str
    as_of: datetime
    sources: list[str] = Field(description="Where the records came from: imported, tawos or synthetic")
    closed_sprints: int
    cold_start: bool = Field(description="Fewer closed sprints than the models need to trust the team history")
    sprints_needed: int = Field(description="Closed sprints still needed to leave the cold start")
    team_context: TeamContext
    sprints: list[HistorySprint]


class HistoryImport(BaseModel):
    project_id: str
    source: str
    sprints: int
    stories: int
    rows: int


class SprintStory(BaseModel):
    """A story in the sprint, as the platform knows it now."""

    story_id: str
    issue_type: str | None = Field(default=None, description="Story, Task, Bug, Improvement or New Feature")
    committed_at: datetime = Field(description="When it was committed to the sprint (its start for a planned story)")
    left_at: datetime | None = Field(default=None, description="When it was taken out before the end")
    points_at_commit: float | None = Field(default=None, ge=0)
    points_at_close: float | None = Field(default=None, ge=0)
    done_in_sprint: bool | None = Field(default=None, description="Finished in this sprint (known once it closes)")
    spilled_over: bool | None = Field(default=None, description="Not done by the end of the story's first sprint "
                                                                "(R1); left empty, the service derives it at the close")
    reopened: bool | None = Field(default=None, description="Reopened after done, then or in the next sprint (R6)")
    started_at: datetime | None = None
    resolved_at: datetime | None = None
    hours_in_progress: float | None = Field(default=None, ge=0)


class SprintRecord(BaseModel):
    """A sprint as the platform knows it now. Send it whenever it changes (it starts, a story is added, taken out
    or done, it closes): it replaces the service's copy of this sprint."""

    name: str | None = None
    started_at: datetime
    planned_end: datetime
    closed_at: datetime | None = Field(default=None, description="Empty while the sprint runs")
    stories: list[SprintStory] = Field(default=[], max_length=2000)


class SprintUpdate(BaseModel):
    project_id: str
    sprint_id: str
    closed: bool
    stories: int
    outcomes_recorded: int = Field(description="Outcomes (FR19) recorded or updated for the stories' predictions")


class SprintRemoved(BaseModel):
    project_id: str
    sprint_id: str
    removed: bool


# ---------------------------------------------------------------------------------------------- read-back


class RecordedFeedback(Feedback):
    id: str
    created_at: datetime


class RecordedOutcome(Outcome):
    id: str
    created_at: datetime


class OutcomeBrief(BaseModel):
    completed_in_sprint: bool
    actual_story_points: float | None = None
    reopened: bool = False


class PredictionBrief(BaseModel):
    """A past prediction in brief, with the latest decision on it and what happened."""

    prediction_id: str
    created_at: datetime
    story_id: str
    sprint_id: str | None = None
    configuration_id: str
    model_version: str
    selection_mode: Literal["auto", "pinned"]
    predicted_story_points: float
    prediction_interval: PredictionInterval
    effort_category: str
    spillover_probability: float
    sprint_risk_level: Literal["low", "medium", "high"]
    confidence_score: float
    confidence_level: str
    feedback: Literal["accept", "adjust", "reject"] | None = Field(default=None, description="The latest decision")
    outcome: OutcomeBrief | None = Field(default=None, description="What happened, once known")


class PredictionPage(BaseModel):
    project_id: str
    predictions: list[PredictionBrief] = Field(description="Newest first")
    next_offset: int | None = Field(default=None, description="Pass as offset for the next page; empty at the end")


class PredictionDetail(BaseModel):
    """One prediction as it was sent, what the models saw (FR21), and what was decided and happened (FR19)."""

    prediction_id: str
    project_id: str
    sprint_id: str | None = None
    created_at: datetime
    prediction: Prediction
    features: dict = Field(description="The feature values the models saw")
    feedback: list[RecordedFeedback]
    outcomes: list[RecordedOutcome]


class ProjectSummary(BaseModel):
    """A project's predictions at a glance, and how they turned out so far."""

    project_id: str
    sprint_id: str | None = None
    predictions: int
    stories: int
    first_at: datetime | None = None
    last_at: datetime | None = None
    by_risk_level: dict[str, int]
    by_configuration: dict[str, int]
    pinned: int = Field(description="Predictions from a configuration the product owner pinned")
    feedback: dict[str, int] = Field(description="Decisions: accept, adjust, reject")
    outcomes: int = Field(description="Predictions whose outcome is known")
    completed_share: float | None = Field(default=None, description="Share of those stories done in their sprint")
    mean_spillover_probability: float | None = Field(
        default=None, description="What the models predicted for those stories, to compare with 1 - completed_share")
    effort_mae: float | None = Field(default=None, description="Mean absolute error against the actual points")
