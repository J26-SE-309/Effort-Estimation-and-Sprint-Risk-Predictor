"""The link between the API and the prediction engine (erp.serving in ml-engine/).

The engine is created once, at start-up, from the arena's models (settings.models_dir); every request reuses
it and its loaded models (NFR1: models are loaded at start-up, never per request).
"""

from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, status

from app.schemas import EstimateRequest


@lru_cache
def load_engine(models_dir: Path, preload: bool = True):
    from erp.serving.engine import Engine

    return Engine(models_dir, preload=preload)


def get_engine():
    from app.config import get_settings

    settings = get_settings()
    try:
        return load_engine(settings.models_dir, settings.preload_models)
    except FileNotFoundError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"No trained models at {settings.models_dir}") \
            from error


def engine_arguments(request: EstimateRequest) -> dict:
    """The request as the engine takes it: plain dicts with the API's field names. Team and sprint context the
    caller left out come from the project's sprint history (app.history), when it has one."""
    from app import history

    stories = [story.model_dump() for story in request.stories]
    team, sprint = history.context(request.project_id, request.sprint_id, frozenset(s["story_id"] for s in stories))
    return {
        "project_id": request.project_id,
        "stories": stories,
        "team": history.merged(request.team_context.model_dump() if request.team_context else None, team,
                               "velocity_mean"),
        "sprint_context": history.merged(request.sprint_context.model_dump() if request.sprint_context else None,
                                         sprint, "length_days"),
    }

