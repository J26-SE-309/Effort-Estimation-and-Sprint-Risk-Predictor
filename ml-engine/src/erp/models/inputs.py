"""Turning a story's features and text vector into model inputs, identically in training and in a saved bundle."""

import numpy as np
import pandas as pd

from erp.features import catalog
from erp.models import encoders

CATEGORICAL = ["issue_type", "priority_level", "project_key"]


def structured(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    """The catalogue features as model inputs. M1 never sees story_points: they are its answer."""
    names = [n for n in catalog.names() if not (task == "effort" and n == "story_points")]
    x = frame[names].copy()
    for column in x.columns:
        if column in CATEGORICAL:
            x[column] = pd.Categorical(x[column].astype(str))
        elif pd.api.types.is_bool_dtype(x[column]):
            x[column] = x[column].astype(int)
    return x


def design(frame: pd.DataFrame, text: np.ndarray | None, task: str, use_text: bool = True,
           use_features: bool = True) -> pd.DataFrame:
    parts = []
    if use_text:
        parts.append(pd.DataFrame(text, index=frame.index, columns=encoders.SbertEncoder().columns()))
    if use_features:
        parts.append(structured(frame, task))
    return pd.concat(parts, axis=1)
