"""The Comparative Model Arena's configurations, fixed before any result was seen (proposal Appendix D).

Fixing the list in advance is the guard against 'with dozens of configurations one will win by chance'
(ML guide 10.1): nothing is added or dropped after looking at the test split. Every configuration gets the same
tuning budget, the same inner folds, the same seeds and the same data.
"""

from dataclasses import dataclass

ARENA = "arena-v1"
SEED = 42
N_TRIALS = 25  # Optuna trials per configuration and task, the same budget for every configuration
INNER_FOLDS = 3


@dataclass(frozen=True)
class Config:
    id: str
    label: str
    encoder: str  # tfidf, fasttext, sbert, distilbert, or "several" for the stack
    learner: str  # a key of erp.models.learners.LEARNERS, or mlp / distilbert / stack
    role: str  # its distinct role in the arena, from Appendix D

    @property
    def joint(self) -> bool:
        return self.learner == "mlp"


CONFIGS = [
    Config("tfidf-rf", "TF-IDF + Random Forest", "tfidf", "random_forest", "Interpretable lexical baseline"),
    Config("tfidf-svm", "TF-IDF + SVR / SVM", "tfidf", "svm", "Classical baseline used in prior literature"),
    Config("fasttext-lightgbm", "FastText + LightGBM", "fasttext", "lightgbm", "Dense embeddings at low cost"),
    Config("sbert-xgboost", "SBERT + XGBoost", "sbert", "xgboost", "Semantic encoding with a strong tabular learner"),
    Config("sbert-lightgbm", "SBERT + LightGBM", "sbert", "lightgbm", "Faster variant on wide feature matrices"),
    Config("sbert-catboost", "SBERT + CatBoost", "sbert", "catboost",
           "Strong handling of categorical project metadata"),
    Config("sbert-mtl", "SBERT + multi-task MLP (M3)", "sbert", "mlp", "Tests the joint effort-and-risk hypothesis"),
    Config("distilbert", "Fine-tuned DistilBERT", "distilbert", "distilbert",
           "Upper bound on contextual representation"),
    Config("stack", "Stacked ensemble", "several", "stack", "Combines complementary learner strengths"),
]
BY_ID = {c.id: c for c in CONFIGS}
TRAINED_BY_ARENA = [c.id for c in CONFIGS if c.learner not in ("distilbert", "stack")]
STACK_BASES = [c.id for c in CONFIGS if c.learner not in ("mlp", "distilbert", "stack")]
STACK_SIZE = 3  # the best three single-task configurations (by their inner-fold score) for each task
