"""Turning a story's features and text vector into model inputs, identically in training and in a saved model.

Tree learners that understand categories (LightGBM, XGBoost, CatBoost) get a table with category columns.
Learners that need plain numbers (Random Forest, SVR / SVM, the MLP) get the same table through TabularPrep:
one column per category value, missing values filled with the training median plus a 'was missing' flag, and
every column scaled with the training mean and spread. TabularPrep is fitted on training rows only and saved as
JSON with the model.
"""

import numpy as np
import pandas as pd

from erp.features import catalog

CATEGORICAL = ["issue_type", "priority_level", "project_key"]
MISSING_SUFFIX = "__missing"


def structured(frame: pd.DataFrame, task: str, levels: dict[str, list[str]] | None = None,
               drop: tuple[str, ...] = ()) -> pd.DataFrame:
    """The catalogue features as model inputs. M1 (and M3) never see story_points: they are its answer.

    levels fixes the category values (and their order), so a model sees the same codes in training and later.
    drop leaves out features (the H2 ablations).
    """
    names = [n for n in catalog.names() if n not in drop and not (task in ("effort", "joint") and n == "story_points")]
    x = frame[names].copy()
    for column in x.columns:
        if column in CATEGORICAL:
            values = x[column].astype(str)
            if levels:  # a value the models never saw (a new project, a missing type) is unknown: NaN
                values = values.where(values.isin(levels[column]))
            x[column] = pd.Categorical(values, categories=levels[column] if levels else None)
        elif pd.api.types.is_bool_dtype(x[column]):
            x[column] = x[column].astype(int)
    return x


def category_levels(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(frame[c].astype(str).unique().tolist()) for c in CATEGORICAL}


def design(frame: pd.DataFrame, text: np.ndarray | None, task: str, use_text: bool = True,
           use_features: bool = True, text_columns: list[str] | None = None,
           levels: dict[str, list[str]] | None = None, drop: tuple[str, ...] = ()) -> pd.DataFrame:
    """Text vector columns (named after the encoder, e.g. sbert_0) followed by the structured features."""
    parts = []
    if use_text:
        if text_columns is None:
            from erp.models.encoders import SbertEncoder

            text_columns = SbertEncoder().columns()
        parts.append(pd.DataFrame(text, index=frame.index, columns=text_columns))
    if use_features:
        parts.append(structured(frame, task, levels, drop))
    return pd.concat(parts, axis=1)


class TabularPrep:
    """One-hot categories, median-fill with 'was missing' flags, and standard scaling; fitted on training rows."""

    def __init__(self, scale: bool = True):
        self.scale = scale
        self.categories: dict[str, list[str]] = {}
        self.numeric: list[str] = []
        self.medians: dict[str, float] = {}
        self.flagged: list[str] = []
        self.mean: list[float] = []
        self.std: list[float] = []

    def fit(self, x: pd.DataFrame) -> "TabularPrep":
        self.categories = {c: [str(v) for v in x[c].cat.categories] if hasattr(x[c], "cat")
                           else sorted(x[c].astype(str).unique().tolist())
                           for c in x.columns if c in CATEGORICAL}
        self.numeric = [c for c in x.columns if c not in CATEGORICAL]
        numbers = x[self.numeric].astype(float)
        self.medians = {c: float(v) if pd.notna(v) else 0.0 for c, v in numbers.median().items()}
        self.flagged = [c for c in self.numeric if numbers[c].isna().any()]
        self.mean, self.std = [], []
        matrix = self._raw(x)
        if self.scale:
            self.mean = matrix.mean(axis=0).tolist()
            self.std = np.where(matrix.std(axis=0) > 1e-12, matrix.std(axis=0), 1.0).tolist()
        return self

    def _raw(self, x: pd.DataFrame) -> np.ndarray:
        numbers = x[self.numeric].astype(float)
        parts = [numbers.fillna(self.medians).to_numpy(np.float64)]
        if self.flagged:
            parts.append(numbers[self.flagged].isna().to_numpy(np.float64))
        for column, values in self.categories.items():
            as_text = x[column].astype(str).to_numpy()
            parts.append(np.stack([as_text == v for v in values], axis=1).astype(np.float64))
        return np.hstack(parts)

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        matrix = self._raw(x)
        if self.scale:
            matrix = (matrix - np.asarray(self.mean)) / np.asarray(self.std)
        return matrix.astype(np.float32)

    def columns(self) -> list[str]:
        return (self.numeric + [c + MISSING_SUFFIX for c in self.flagged]
                + [f"{c}={v}" for c, values in self.categories.items() for v in values])

    def to_dict(self) -> dict:
        return {"scale": self.scale, "categories": self.categories, "numeric": self.numeric,
                "medians": self.medians, "flagged": self.flagged, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, data: dict) -> "TabularPrep":
        prep = cls(data["scale"])
        for key in ("categories", "numeric", "medians", "flagged", "mean", "std"):
            setattr(prep, key, data[key])
        return prep
