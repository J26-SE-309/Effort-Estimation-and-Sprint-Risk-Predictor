"""What every arena configuration shares: the stories, the splits, the inner folds and the encoded text.

Inner folds (time-aware nested cross-validation, ML guide 8.1): inside the training split, each project's
sprints are ordered by start date and cut into INNER_FOLDS + 1 blocks of consecutive sprints. Fold i trains on
blocks 1..i and validates on block i + 1, so tuning never uses a later sprint to predict an earlier one, and
the calibration and test splits are never touched.

E1 and E2 learn from text, so for every fold they are fitted on that fold's training stories only, and for the
final models on the whole training split. Encoded matrices are data (they encode TAWOS stories) and are cached
in the Datasets folder; the fitted encoders are models and are saved in the repository with the arena.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from erp import config
from erp.arena.configs import ARENA, INNER_FOLDS
from erp.models import encoders
from erp.models.first_models import load_dataset
from erp.models.inputs import category_levels

MODELS_DIR = config.MODEL_BUNDLES_DIR / ARENA
WORK_DIR = config.WORK_DIR / "arena" / ARENA
ENCODERS_DIR = MODELS_DIR / "encoders"


def inner_folds(train: pd.DataFrame, k: int = INNER_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """(fit, validate) boolean masks over the rows of train, forward-chaining over each project's sprints."""
    block = pd.Series(-1, index=train.index)
    for _, group in train.groupby("Project_ID"):
        order = group.sort_values(["sprint_start", "Sprint_ID"])["Sprint_ID"].drop_duplicates().to_numpy()
        for number, sprints in enumerate(np.array_split(order, k + 1)):
            block[group.index[group["Sprint_ID"].isin(sprints)]] = number
    blocks = block.to_numpy()
    return [(blocks < i, blocks == i) for i in range(1, k + 1)]


def targets(frame: pd.DataFrame, task: str) -> np.ndarray:
    """Effort: log(1 + story points) (right-skewed, ML guide 4.1). Risk: at_risk as 0 / 1."""
    if task == "effort":
        return np.log1p(frame["story_points"].to_numpy(float))
    return frame["at_risk"].astype(int).to_numpy()


def _save_array(path: Path, array: np.ndarray, key: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, array)
    temporary.replace(path)  # atomic, so two arena processes never read a half-written file
    path.with_suffix(".json").write_text(json.dumps(key), encoding="utf-8")


def _load_array(path: Path, key: dict) -> np.ndarray | None:
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == key:
        return np.load(path, allow_pickle=False)
    return None


@dataclass
class ArenaData:
    frame: pd.DataFrame
    texts: pd.Series
    levels: dict
    folds: list = field(default_factory=list)

    @classmethod
    def load(cls, interim: Path | None = None) -> "ArenaData":
        frame = load_dataset(interim or config.INTERIM_DIR)
        data = cls(frame=frame, texts=encoders.story_text(frame), levels=category_levels(frame))
        data.folds = inner_folds(data.train)
        return data

    @property
    def parts(self) -> pd.Series:
        return self.frame["split"]

    def mask(self, *names: str) -> np.ndarray:
        return self.parts.isin(names).to_numpy()

    @property
    def train(self) -> pd.DataFrame:
        return self.frame[self.mask("train")]

    def _key(self, encoder_name: str, rows: pd.Index) -> dict:
        ids = hashlib.sha1(",".join(map(str, rows)).encode()).hexdigest()
        params = getattr(encoders.ENCODERS[encoder_name], "params", {})
        return {"encoder": encoder_name, "params": params, "fitted_on": ids}

    def fold_text(self, encoder_name: str, fold: int) -> np.ndarray:
        """Every training story encoded by an encoder fitted on the fold's own training stories."""
        if encoder_name == "sbert":
            return self.full_text("sbert")[self.mask("train")]
        train = self.train
        fit_rows = train.index[self.folds[fold][0]]
        path = WORK_DIR / "folds" / f"{encoder_name}-fold{fold + 1}.npy"
        key = {**self._key(encoder_name, fit_rows), "encoded": self._key(encoder_name, train.index)["fitted_on"]}
        cached = _load_array(path, key)
        if cached is None:
            texts = self.texts[self.mask("train")]
            encoder = encoders.ENCODERS[encoder_name]().fit(texts[self.folds[fold][0]])
            cached = encoder.transform(texts)
            _save_array(path, cached, key)
        return cached

    def full_text(self, encoder_name: str) -> np.ndarray:
        """Every story encoded by the encoder fitted on the whole training split (saved in the repository)."""
        if encoder_name == "sbert":
            return encoders.SbertEncoder().transform(self.texts)
        path = WORK_DIR / "encoded" / f"{encoder_name}.npy"
        key = {**self._key(encoder_name, self.train.index), "encoded": self._key(encoder_name, self.frame.index)[
            "fitted_on"]}
        cached = _load_array(path, key)
        saved = ENCODERS_DIR / encoder_name / "encoder.json"
        if cached is None or not saved.exists():
            encoder = encoders.ENCODERS[encoder_name]().fit(self.texts[self.mask("train")])
            spec = encoder.save(ENCODERS_DIR / encoder_name)
            saved.write_text(json.dumps({**spec, "fitted_on": key["fitted_on"]}, indent=1), encoding="utf-8")
            cached = encoder.transform(self.texts)
            _save_array(path, cached, key)
        return cached


def load_encoder(name: str, directory: Path = ENCODERS_DIR):
    """The fitted encoder shared by an arena's configurations: `directory` is that arena's encoders folder."""
    if name == "sbert":
        return encoders.SbertEncoder()
    return encoders.ENCODERS[name].load(directory / name)


def text_columns(name: str) -> list[str]:
    return encoders.ENCODERS[name]().columns()
