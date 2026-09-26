"""Version 1 of the prediction API (proposal Figure 2: /estimate, /risk, /recommend, /models, /compare).

Predictions come from the Comparative Model Arena's configurations (ml-engine/models/arena-v1) through the
prediction engine in ml-engine (erp.serving): the router picks the configuration (FR11) unless the product owner
pinned one (FR12), and every prediction carries its interval and calibrated probability (FR13), its reasons
(FR14), its recommendations (FR15), the feature groups it used (FR17) and its model version (FR21).

The prediction endpoints are CPU-bound and deliberately run on the worker's event loop (`async def` without an
await): a busy worker process then accepts no new connection, and the kernel hands it to an idle one. As plain
`def` endpoints they ran in a thread pool while the free event loop kept accepting, so one worker took nearly all
concurrent requests and the others idled (NFR1 missed: p95 2.3 s with 4 workers on Linux).
"""

from fastapi import APIRouter, HTTPException, status

from app import store
from app.prediction import engine_arguments, get_engine
from app.schemas import (
    CompareRequest,
    CompareResponse,
    EstimateRequest,
    EstimateResponse,
    Feedback,
    ModelConfiguration,
    ModelsResponse,
    ModelSummary,
    Outcome,
    Pin,
    PinRequest,
    RecommendResponse,
    Recorded,
    SprintRiskResponse,
    StoryRecommendations,
)

router = APIRouter(tags=["effort estimation and sprint risk"])


def _pinned(request: EstimateRequest) -> str | None:
    """The request's own pin, else the one stored for the project."""
    return request.pinned_configuration or store.get_pin(request.project_id)[0]


def _run(request: EstimateRequest, sprint_level: bool = False) -> dict:
    engine = get_engine()
    method = engine.sprint_risk if sprint_level else engine.estimate
    try:
        result = method(**engine_arguments(request), pinned=_pinned(request))
    except KeyError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error.args[0])) from error
    features = result.pop("features")
    store.record_predictions(request.project_id, request.sprint_id, result["predictions"], features)
    return result


@router.post("/estimate", response_model=EstimateResponse)
async def estimate(request: EstimateRequest) -> dict:
    """Estimate effort and sprint risk for every story in a backlog, in the order given."""
    return _run(request)


@router.post("/risk", response_model=SprintRiskResponse)
async def sprint_risk(request: EstimateRequest) -> dict:
    """Sprint-level risk of committing to this backlog (FR16, Monte Carlo), with the story predictions.

    Over-commitment needs the team's capacity: sprint_context.capacity_points or team_context.velocity_mean.
    """
    return _run(request, sprint_level=True)


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: EstimateRequest) -> dict:
    """Planning recommendations for a backlog (FR15): per story, and for the sprint as a whole."""
    result = _run(request, sprint_level=True)
    return {
        "stories": [StoryRecommendations(**{k: p[k] for k in StoryRecommendations.model_fields})
                    for p in result["predictions"]],
        "sprint": result["sprint"]["recommendations"],
    }


@router.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest) -> dict:
    """The same backlog through several configurations side by side (FR12). Not recorded in the audit log."""
    arguments = engine_arguments(request)
    try:
        result = get_engine().compare(arguments["project_id"], arguments["stories"], request.configurations,
                                      arguments["team"], arguments["sprint_context"])
    except KeyError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error.args[0])) from error
    return {"results": [{"configuration_id": r["configuration_id"], "predictions": r["predictions"]}
                        for r in result["results"]]}


@router.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    """The arena's configurations with their test-split leaderboard metrics (FR10)."""
    from erp.arena import configs

    board = get_engine().models()
    listed = {m["configuration_id"] for m in board["configurations"]}
    summaries = [ModelSummary(
        configuration_id=m["configuration_id"], label=m["label"], role=m["role"],
        configuration=ModelConfiguration(encoder=m["encoder"], learner=m["learner"],
                                         formulation="multi_task" if m["learner"] == "mlp" else "single_task"),
        status=m["status"], loaded=m["loaded"], eligible=m["eligible"],
        failed_requirements=m["failed_requirements"], composite=m["composite"], metrics=m["metrics"])
        for m in board["configurations"]]
    summaries += [ModelSummary(  # in the arena's plan but not trained (DistilBERT waits for its GPU run)
        configuration_id=c.id, label=c.label, role=c.role, status="planned", loaded=False,
        configuration=ModelConfiguration(encoder=c.encoder, learner=c.learner, formulation="single_task"))
        for c in configs.CONFIGS if c.id not in listed]
    return ModelsResponse(arena=board["arena"], pooled_winner=board["pooled_winner"], weights=board["weights"],
                          configurations=summaries)


# ------------------------------------------------------------------ pinning (FR12)


@router.get("/projects/{project_id}/pin", response_model=Pin)
def get_pin(project_id: str) -> Pin:
    configuration_id, pinned_at = store.get_pin(project_id)
    return Pin(project_id=project_id, configuration_id=configuration_id, pinned_at=pinned_at)


@router.put("/projects/{project_id}/pin", response_model=Pin)
def pin(project_id: str, request: PinRequest) -> Pin:
    """Answer this project's requests with one configuration from now on, instead of the router's choice."""
    if request.configuration_id not in get_engine().router.available:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            f"configuration {request.configuration_id!r} is not available")
    if not store.set_pin(project_id, request.configuration_id):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The database is unavailable; nothing was pinned")
    return get_pin(project_id)


@router.delete("/projects/{project_id}/pin", response_model=Pin)
def unpin(project_id: str) -> Pin:
    if not store.set_pin(project_id, None):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The database is unavailable; nothing changed")
    return Pin(project_id=project_id, configuration_id=None)


# ------------------------------------------------------------------ feedback and outcomes (FR19)


def _known(prediction_id: str) -> None:
    if store.prediction_exists(prediction_id) is False:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No prediction {prediction_id}")


@router.post("/feedback", response_model=Recorded, status_code=status.HTTP_201_CREATED)
def feedback(request: Feedback) -> Recorded:
    """Record a product owner's accept / adjust / reject decision on a prediction or a recommendation."""
    _known(request.prediction_id)
    record_id, recorded = store.record_feedback(request.model_dump())
    return Recorded(id=record_id, recorded=recorded)


@router.post("/outcomes", response_model=Recorded, status_code=status.HTTP_201_CREATED)
def outcome(request: Outcome) -> Recorded:
    """Record what really happened to the story in its sprint, for evaluation and retraining (FR20)."""
    _known(request.prediction_id)
    record_id, recorded = store.record_outcome(request.model_dump())
    return Recorded(id=record_id, recorded=recorded)
