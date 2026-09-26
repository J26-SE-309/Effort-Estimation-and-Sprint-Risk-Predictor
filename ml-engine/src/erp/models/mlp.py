"""Feed-forward networks in PyTorch: M3, the joint multi-task network, and its single-task twins (ML guide 4.3).

One architecture for all three, so the H1 comparison is fair: a shared trunk of two dense layers (256 -> 128,
layer normalisation, ReLU, dropout) and one output head per task. The effort head predicts standardised
log(1 + story points) with a Huber loss; the risk head predicts a logit with binary cross-entropy (the positive
class weighted by the training imbalance). The joint network learns how to weight its two losses itself
(uncertainty weighting, Kendall et al., 2018): total = sum over tasks of exp(-s) x loss + s, with s learned.

Inputs are the text vector plus the structured features through TabularPrep. story_points is never an input:
it is the effort head's answer, and the single-task risk network leaves it out too, so all three networks see
exactly the same inputs. Training stops early on the validation loss (the calibration split in the final fit,
as ML guide 8.1 allows for neural networks) and keeps the best epoch.

Saved as safetensors (weights only, cannot run code) plus JSON with the architecture, TabularPrep and target
scaling.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from erp.models.inputs import TabularPrep

SEED = 42
TRUNK = (256, 128)
MAX_EPOCHS = 200
PATIENCE = 20
HUBER_DELTA = 1.0


def _network(inputs: int, tasks: tuple[str, ...], dropout: float):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers, width = [], inputs
            for units in TRUNK:
                layers += [nn.Linear(width, units), nn.LayerNorm(units), nn.ReLU(), nn.Dropout(dropout)]
                width = units
            self.trunk = nn.Sequential(*layers)
            self.heads = nn.ModuleDict({task: nn.Linear(width, 1) for task in tasks})
            self.log_vars = nn.Parameter(torch.zeros(len(tasks)))  # learned task weights; unused with one task

        def forward(self, x):
            hidden = self.trunk(x)
            return {task: head(hidden).squeeze(-1) for task, head in self.heads.items()}

    return Net()


class MLP:
    """tasks: ('effort',), ('risk',) or ('effort', 'risk') for M3."""

    name, label = "mlp", "MLP"

    def __init__(self, tasks: tuple[str, ...]):
        self.tasks = tuple(tasks)
        self.prep: TabularPrep | None = None
        self.net = None
        self.params: dict = {}
        self.target: dict = {}
        self.pos_weight = 1.0
        self.best_epoch = 0
        self.best_val_loss = float("inf")

    @staticmethod
    def space(trial) -> dict:
        return {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }

    def _tensors(self, x: pd.DataFrame, y: dict | None):
        import torch

        features = torch.from_numpy(self.prep.transform(x))
        if y is None:
            return features, None
        targets = {}
        if "effort" in self.tasks:
            targets["effort"] = torch.from_numpy(
                ((np.asarray(y["effort"], float) - self.target["mean"]) / self.target["std"]).astype(np.float32))
        if "risk" in self.tasks:
            targets["risk"] = torch.from_numpy(np.asarray(y["risk"], np.float32))
        return features, targets

    def _losses(self, out: dict, targets: dict) -> dict:
        import torch
        from torch.nn import functional

        losses = {}
        if "effort" in self.tasks:
            losses["effort"] = functional.huber_loss(out["effort"], targets["effort"], delta=HUBER_DELTA)
        if "risk" in self.tasks:
            losses["risk"] = functional.binary_cross_entropy_with_logits(
                out["risk"], targets["risk"], pos_weight=torch.tensor(self.pos_weight))
        return losses

    def _total(self, losses: dict):
        if len(self.tasks) == 1:
            return losses[self.tasks[0]]
        import torch

        return sum(torch.exp(-self.net.log_vars[i]) * losses[t] + self.net.log_vars[i]
                   for i, t in enumerate(self.tasks))

    def fit(self, x: pd.DataFrame, y: dict, x_val: pd.DataFrame, y_val: dict, params: dict,
            seed: int = SEED) -> "MLP":
        """y and y_val: {'effort': log(1 + points), 'risk': 0 / 1} for the network's tasks."""
        import torch

        self.params = dict(params)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        self.prep = TabularPrep(scale=True).fit(x)
        if "effort" in self.tasks:
            effort = np.asarray(y["effort"], float)
            self.target = {"mean": float(effort.mean()), "std": float(effort.std())}
        if "risk" in self.tasks:
            share = float(np.mean(y["risk"]))
            self.pos_weight = (1 - share) / share
        features, targets = self._tensors(x, y)
        val_features, val_targets = self._tensors(x_val, y_val)
        self.net = _network(features.shape[1], self.tasks, params["dropout"])
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

        best_state, best_loss, best_epoch, waited = None, float("inf"), 0, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            self.net.train()
            order = torch.randperm(len(features), generator=generator)
            for start in range(0, len(order), params["batch_size"]):
                batch = order[start:start + params["batch_size"]]
                optimiser.zero_grad()
                losses = self._losses(self.net(features[batch]), {t: v[batch] for t, v in targets.items()})
                self._total(losses).backward()
                optimiser.step()
            self.net.eval()
            with torch.no_grad():
                # stop on the plain sum of the task losses: the learned weights move, so their sum would too
                val_loss = float(sum(self._losses(self.net(val_features), val_targets).values()))
            if val_loss < best_loss - 1e-6:
                best_loss, best_epoch, waited = val_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.best_epoch, self.best_val_loss = best_epoch, best_loss
        return self

    def predict(self, x: pd.DataFrame) -> dict[str, np.ndarray]:
        """{'effort': log(1 + points), 'risk': probability before C1} for the network's tasks."""
        import torch

        features, _ = self._tensors(x, None)
        with torch.no_grad():
            out = self.net(features)
        result = {}
        if "effort" in self.tasks:
            result["effort"] = out["effort"].numpy().astype(float) * self.target["std"] + self.target["mean"]
        if "risk" in self.tasks:
            result["risk"] = torch.sigmoid(out["risk"]).numpy().astype(float)
        return result

    def save(self, directory: Path, prefix: str) -> dict:
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in self.net.state_dict().items()}, directory / f"{prefix}.safetensors")
        return {"file": f"{prefix}.safetensors", "tasks": list(self.tasks), "trunk": list(TRUNK),
                "inputs": int(len(self.prep.columns())), "params": self.params, "target": self.target,
                "pos_weight": self.pos_weight, "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss, "prep": self.prep.to_dict()}

    @classmethod
    def load(cls, directory: Path, spec: dict) -> "MLP":
        from safetensors.torch import load_file

        model = cls(tuple(spec["tasks"]))
        model.prep = TabularPrep.from_dict(spec["prep"])
        model.params, model.target, model.pos_weight = spec["params"], spec["target"], spec["pos_weight"]
        model.best_epoch, model.best_val_loss = spec["best_epoch"], spec["best_val_loss"]
        model.net = _network(spec["inputs"], model.tasks, spec["params"]["dropout"])
        model.net.load_state_dict(load_file(directory / spec["file"]))
        model.net.eval()
        return model
