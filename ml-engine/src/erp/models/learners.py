"""Learners behind one fit() / predict() / save() / load() interface (ML guide 3.2 step 6, NFR9).

Every learner predicts log(1 + story points) for effort and a score between 0 and 1 for risk (the SVM's
margin goes through a sigmoid; C1 then turns any score into an honest probability). Effort uses absolute or
absolute-like losses so a few 40-point stories cannot dominate (ML guide 4.1): L1 for LightGBM and XGBoost, MAE
for CatBoost, epsilon-insensitive for SVR. Random Forest uses squared error on the log scale, because sklearn's
absolute-error trees are far too slow at this size; the log already tames the big stories.

Saved in each library's own format, never as pickles: LightGBM text, XGBoost UBJSON, CatBoost .cbm, and
scikit-learn models (Random Forest, SVR / SVM) with skops, which refuses to load anything but the scikit-learn
and NumPy types it is told to trust.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from erp.models.inputs import CATEGORICAL, TabularPrep

SEED = 42
MAX_ROUNDS = 3000
EARLY_STOPPING = 100


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(z, float)))


class LightGBMLearner:
    name, label = "lightgbm", "LightGBM"

    def __init__(self, task: str):
        self.task = task
        self.booster = None
        self.params: dict = {}
        self.best_iteration = 0

    @staticmethod
    def space(trial) -> dict:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
        }

    def fit(self, x, y, x_val, y_val, params: dict, seed: int = SEED) -> "LightGBMLearner":
        import lightgbm as lgb

        self.params = dict(params)
        common = {"n_estimators": MAX_ROUNDS, "subsample_freq": 1, "random_state": seed, "verbose": -1, **params}
        model = (lgb.LGBMRegressor(objective="l1", **common) if self.task == "effort"
                 else lgb.LGBMClassifier(objective="binary", **common))
        model.fit(x, y, eval_X=(x_val,), eval_y=(y_val,),
                  callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
        self.booster = model.booster_
        self.best_iteration = int(model.best_iteration_ or MAX_ROUNDS)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(x), float)

    def contributions(self, x: pd.DataFrame) -> pd.DataFrame:
        values = self.booster.predict(x, pred_contrib=True)
        return pd.DataFrame(np.asarray(values)[:, :-1], index=x.index, columns=x.columns)

    def save(self, directory: Path, prefix: str) -> dict:
        self.booster.save_model(directory / f"{prefix}.txt")
        return {"file": f"{prefix}.txt", "params": self.params, "best_iteration": self.best_iteration}

    @classmethod
    def load(cls, directory: Path, task: str, spec: dict) -> "LightGBMLearner":
        import lightgbm as lgb

        learner = cls(task)
        learner.booster = lgb.Booster(model_file=str(directory / spec["file"]))
        learner.params, learner.best_iteration = spec["params"], spec["best_iteration"]
        return learner


class XGBoostLearner:
    name, label = "xgboost", "XGBoost"

    def __init__(self, task: str):
        self.task = task
        self.booster = None
        self.params: dict = {}
        self.best_iteration = 0

    @staticmethod
    def space(trial) -> dict:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.8),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        }

    def fit(self, x, y, x_val, y_val, params: dict, seed: int = SEED) -> "XGBoostLearner":
        import xgboost as xgb

        self.params = dict(params)
        common = {"n_estimators": MAX_ROUNDS, "tree_method": "hist", "enable_categorical": True,
                  "early_stopping_rounds": EARLY_STOPPING, "random_state": seed, "n_jobs": -1, **params}
        model = (xgb.XGBRegressor(objective="reg:absoluteerror", **common) if self.task == "effort"
                 else xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common))
        model.fit(x, y, eval_set=[(x_val, y_val)], verbose=False)
        self.best_iteration = int(model.best_iteration) + 1
        self.booster = model.get_booster()[: self.best_iteration]  # keep only the trees that are used
        return self

    def _matrix(self, x: pd.DataFrame):
        import xgboost as xgb

        return xgb.DMatrix(x, enable_categorical=True)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(self._matrix(x)), float)

    def contributions(self, x: pd.DataFrame) -> pd.DataFrame:
        values = self.booster.predict(self._matrix(x), pred_contribs=True)
        return pd.DataFrame(np.asarray(values)[:, :-1], index=x.index, columns=x.columns)

    def save(self, directory: Path, prefix: str) -> dict:
        self.booster.save_model(directory / f"{prefix}.ubj")
        return {"file": f"{prefix}.ubj", "params": self.params, "best_iteration": self.best_iteration}

    @classmethod
    def load(cls, directory: Path, task: str, spec: dict) -> "XGBoostLearner":
        import xgboost as xgb

        learner = cls(task)
        learner.booster = xgb.Booster(model_file=str(directory / spec["file"]))
        learner.params, learner.best_iteration = spec["params"], spec["best_iteration"]
        return learner


class CatBoostLearner:
    name, label = "catboost", "CatBoost"

    def __init__(self, task: str):
        self.task = task
        self.model = None
        self.params: dict = {}
        self.best_iteration = 0

    @staticmethod
    def space(trial) -> dict:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "rsm": trial.suggest_float("rsm", 0.1, 0.8),
        }

    @staticmethod
    def _table(x: pd.DataFrame) -> pd.DataFrame:
        return x.assign(**{c: x[c].astype(str) for c in CATEGORICAL if c in x.columns})

    def fit(self, x, y, x_val, y_val, params: dict, seed: int = SEED) -> "CatBoostLearner":
        from catboost import CatBoostClassifier, CatBoostRegressor

        self.params = dict(params)
        common = {"iterations": MAX_ROUNDS, "od_type": "Iter", "od_wait": EARLY_STOPPING, "use_best_model": True,
                  "random_seed": seed, "verbose": False, "thread_count": -1, "allow_writing_files": False, **params}
        model = (CatBoostRegressor(loss_function="MAE", **common) if self.task == "effort"
                 else CatBoostClassifier(loss_function="Logloss", **common))
        cats = [c for c in CATEGORICAL if c in x.columns]
        model.fit(self._table(x), np.asarray(y), cat_features=cats, eval_set=(self._table(x_val), np.asarray(y_val)))
        self.model = model
        self.best_iteration = int(model.get_best_iteration()) + 1
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        table = self._table(x)
        if self.task == "effort":
            return np.asarray(self.model.predict(table), float)
        return np.asarray(self.model.predict_proba(table)[:, 1], float)

    def contributions(self, x: pd.DataFrame) -> pd.DataFrame:
        from catboost import Pool

        cats = [c for c in CATEGORICAL if c in x.columns]
        values = self.model.get_feature_importance(Pool(self._table(x), cat_features=cats), type="ShapValues")
        return pd.DataFrame(np.asarray(values)[:, :-1], index=x.index, columns=x.columns)

    def save(self, directory: Path, prefix: str) -> dict:
        self.model.save_model(str(directory / f"{prefix}.cbm"))
        return {"file": f"{prefix}.cbm", "params": self.params, "best_iteration": self.best_iteration}

    @classmethod
    def load(cls, directory: Path, task: str, spec: dict) -> "CatBoostLearner":
        from catboost import CatBoostClassifier, CatBoostRegressor

        learner = cls(task)
        learner.model = (CatBoostRegressor() if task == "effort" else CatBoostClassifier()).load_model(
            str(directory / spec["file"]))
        learner.params, learner.best_iteration = spec["params"], spec["best_iteration"]
        return learner


TRUSTED_PREFIXES = ("sklearn.", "numpy.", "builtins.")


def _skops_dump(model, path: Path) -> None:
    import skops.io as sio

    sio.dump(model, path, compression=zipfile.ZIP_DEFLATED)


def _skops_load(path: Path):
    """Load a skops file, trusting only scikit-learn, NumPy and built-in types (anything else is refused)."""
    import skops.io as sio

    untrusted = sio.get_untrusted_types(file=path)
    unknown = [t for t in untrusted if not t.startswith(TRUSTED_PREFIXES)]
    if unknown:
        raise ValueError(f"{path.name} contains types that are not trusted: {unknown}")
    return sio.load(path, trusted=untrusted)


class _SklearnLearner:
    """Shared by Random Forest and SVR / SVM: TabularPrep in front, a scikit-learn model behind, saved with skops."""

    scale = True

    def __init__(self, task: str):
        self.task = task
        self.prep: TabularPrep | None = None
        self.model = None
        self.params: dict = {}
        self.best_iteration = 0

    def _make(self, params: dict, seed: int):
        raise NotImplementedError

    def fit(self, x, y, x_val=None, y_val=None, params: dict | None = None, seed: int = SEED):
        self.params = dict(params or {})
        self.prep = TabularPrep(scale=self.scale).fit(x)
        self.model = self._make(self.params, seed).fit(self.prep.transform(x), np.asarray(y))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def save(self, directory: Path, prefix: str) -> dict:
        _skops_dump(self.model, directory / f"{prefix}.skops")
        return {"file": f"{prefix}.skops", "params": self.params, "prep": self.prep.to_dict()}

    @classmethod
    def load(cls, directory: Path, task: str, spec: dict):
        learner = cls(task)
        learner.model = _skops_load(directory / spec["file"])
        learner.prep = TabularPrep.from_dict(spec["prep"])
        learner.params = spec["params"]
        return learner


class RandomForestLearner(_SklearnLearner):
    name, label = "random_forest", "Random Forest"
    scale = False
    n_estimators = 300

    @staticmethod
    def space(trial) -> dict:
        return {
            "max_features": trial.suggest_float("max_features", 0.05, 0.6),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        }

    def _make(self, params: dict, seed: int):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestRegressor if self.task == "effort" else RandomForestClassifier
        return cls(n_estimators=self.n_estimators, n_jobs=-1, random_state=seed, **params)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        # One thread: parallel prediction adds the trees up in a varying order, so the last digits would change
        # from run to run (NFR10 asks for results that regenerate exactly).
        self.model.set_params(n_jobs=1)
        matrix = self.prep.transform(x)
        if self.task == "effort":
            return np.asarray(self.model.predict(matrix), float)
        return np.asarray(self.model.predict_proba(matrix)[:, 1], float)


class SvmLearner(_SklearnLearner):
    """SVR for effort, SVM (SVC) for risk, both with an RBF kernel on scaled inputs."""

    name, label = "svm", "SVR / SVM"

    def __init__(self, task: str):
        super().__init__(task)
        self.label = "SVR" if task == "effort" else "SVM"

    def space(self, trial) -> dict:
        params = {"C": trial.suggest_float("C", 0.03, 30, log=True),
                  "gamma": trial.suggest_float("gamma", 1e-4, 3e-2, log=True)}
        if self.task == "effort":
            params["epsilon"] = trial.suggest_float("epsilon", 0.02, 0.5, log=True)
        return params

    def _make(self, params: dict, seed: int):
        from sklearn.svm import SVC, SVR

        if self.task == "effort":
            return SVR(kernel="rbf", cache_size=2000, **params)
        return SVC(kernel="rbf", cache_size=2000, random_state=seed, **params)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        matrix = self.prep.transform(x)
        if self.task == "effort":
            return np.asarray(self.model.predict(matrix), float)
        return _sigmoid(self.model.decision_function(matrix))


LEARNERS = {cls.name: cls for cls in (LightGBMLearner, XGBoostLearner, CatBoostLearner, RandomForestLearner,
                                      SvmLearner)}
