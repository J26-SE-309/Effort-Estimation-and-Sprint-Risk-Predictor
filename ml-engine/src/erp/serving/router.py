"""R1: which configuration answers for a project (FR11, FR12; ML guide 4.8). Rules, no training.

  1. A configuration pinned by the product owner always answers (selection_mode "pinned").
  2. A team with fewer than MIN_HISTORY_SPRINTS closed sprints (or unknown history) gets the pooled winner.
  3. A project in the arena's per-project leaderboard gets its own winner, but only when that winner beats the
     pooled winner on the project's test stories by at least MARGIN composite points: per-project scores rest
     on few stories, and a tiny lead is noise.
  4. Everyone else gets the pooled winner: the eligible configuration with the best composite score.
Only configurations whose files are present can answer (DistilBERT's weights, for example, stay on the machine
that fine-tuned them), and only when the packages they run on are installed (the service image built without
WITH_TORCH serves no SBERT, multi-task or DistilBERT model, nor the stack built on SBERT ones). The networks
trained in PyTorch run in PyTorch, or without it: SBERT and DistilBERT exported to ONNX (erp.models.onnx_export),
M3 in NumPy. Models are loaded once, on first use, and then kept in memory.
"""

import importlib.util
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from erp.arena import predictor
from erp.models import onnx_runtime
from erp.models.confidence import COLD_START_SPRINTS
from erp.models.encoders import SbertEncoder

MARGIN = 0.02
MIN_HISTORY_SPRINTS = COLD_START_SPRINTS
# Learners that need packages beyond the serving extra (they come with the serving-sbert or serving-onnx extra).
# SBERT and DistilBERT run in PyTorch or from an ONNX export: see _usable.
NEEDS = {"xgboost": ("xgboost",), "catboost": ("catboost",), "mlp": ("safetensors",)}
DISTILBERT_TORCH = ("torch", "transformers", "safetensors")


@dataclass(frozen=True)
class Choice:
    config: str
    mode: str  # "auto" or "pinned"
    reason: str


class Router:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.leaderboard = json.loads((models_dir / "leaderboard.json").read_text(encoding="utf-8"))
        self.available = sorted(p.parent.name for p in models_dir.glob(f"*/{predictor.MANIFEST}")
                                if _usable(p.parent))
        self._loaded: dict = {}
        self._lock = threading.Lock()

    @property
    def pooled_winner(self) -> str:
        winner = self.leaderboard.get("pooled_winner")
        if winner in self.available:
            return winner
        configs = self.leaderboard["configs"]
        ranked = sorted((c for c in self.available if c in configs),
                        key=lambda c: (configs[c]["eligible"], configs[c]["composite"]), reverse=True)
        return ranked[0] if ranked else self.available[0]

    def choose(self, project_key: str, history_sprints: float | None = None, pinned: str | None = None) -> Choice:
        if pinned:
            if pinned not in self.available:
                raise KeyError(f"configuration {pinned!r} is not available; available: {', '.join(self.available)}")
            return Choice(pinned, "pinned", "pinned by the product owner")
        pooled = self.pooled_winner
        if history_sprints is None or history_sprints != history_sprints or history_sprints < MIN_HISTORY_SPRINTS:
            return Choice(pooled, "auto", "pooled winner: the team has fewer than "
                                          f"{MIN_HISTORY_SPRINTS} closed sprints (or its history is unknown)")
        project = self.leaderboard.get("per_project", {}).get(project_key)
        if project and project.get("winner") in self.available and project["winner"] != pooled:
            lead = project["composite"][project["winner"]] - project["composite"].get(pooled, float("-inf"))
            if lead >= MARGIN:
                return Choice(project["winner"], "auto", f"the project's own winner (leads the pooled winner by "
                                                         f"{lead:.3f} on its test stories)")
        reason = ("pooled winner" + ("" if project else ": the project is not in the leaderboard"))
        return Choice(pooled, "auto", reason)

    def load(self, config: str):
        with self._lock:
            if config not in self._loaded:
                self._loaded[config] = predictor.load(self.models_dir / config)
            return self._loaded[config]

    def loaded(self) -> list[str]:
        return sorted(self._loaded)


def _usable(directory: Path) -> bool:
    """True when every model file model.json names is present (local-only weights may be absent elsewhere) and the
    packages the configuration runs on are installed. A stack is usable when all its base configurations are."""
    manifest = json.loads((directory / predictor.MANIFEST).read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    files = [manifest.get(part, {}).get("model", {}).get("file") for part in ("effort", "risk")]
    files.append(manifest.get("joint", {}).get("file"))
    files = [directory / f for f in files if f]
    if config.get("learner") == "distilbert":  # its exported networks, or its weights and PyTorch
        exported = [f.with_suffix(".onnx") for f in files] + [directory / onnx_runtime.TOKENIZER]
        return onnx_runtime.ready(*exported) or (all(f.exists() for f in files) and _installed(*DISTILBERT_TORCH))
    if not all(f.exists() for f in files):
        return False
    bases = {base for part in ("effort", "risk") for base in manifest.get(part, {}).get("bases", [])}
    if bases:
        return all((directory.parent / base / predictor.MANIFEST).exists() and _usable(directory.parent / base)
                   for base in bases)
    if config.get("encoder") == "sbert" and not SbertEncoder.runnable(directory.parent / "encoders" / "sbert"):
        return False
    return _installed(*NEEDS.get(config.get("learner"), ()))


def _installed(*packages: str) -> bool:
    return all(importlib.util.find_spec(package) is not None for package in packages)
