"""A1: the sprint-level risk, by Monte Carlo simulation (FR16; ML guide 4.10).

Does the whole commitment fit the team's capacity? This is judged independently of the stories' own risk
scores. In each of RUNS simulated sprints:
  - every story's effort is drawn around its M1 estimate, on the log scale, with the spread of its C2 interval
    (the 80% interval's half-width is 1.28 standard deviations of a normal distribution);
  - the team's capacity is drawn from its recent velocity (mean and variance);
  - the sprint is over-committed when the drawn effort exceeds the drawn capacity.
The share of over-committed runs is the sprint risk, reported with the expected overflow. Each story also
misses the sprint with its C1 probability, which gives the number of stories likely to spill over.
Stories are treated as independent (a shared blocker would make them fail together; noted as a limitation).
"""

import numpy as np

RUNS = 10_000
SEED = 42
Z80 = 1.2816  # the 80% interval spans +-1.2816 standard deviations of a normal distribution
DEFAULT_CAPACITY_SPREAD = 0.2  # when only the capacity is known: its standard deviation as a share of it
LEVELS = {"medium": 0.3, "high": 0.6}  # the same bands as the story-level risk


def level(probability: float) -> str:
    return "high" if probability >= LEVELS["high"] else "medium" if probability >= LEVELS["medium"] else "low"


def simulate(log_points, high80, probability, capacity_mean: float | None, capacity_variance: float | None = None,
             runs: int = RUNS, seed: int = SEED) -> dict:
    """log_points: M1's log(1 + points); high80: upper end of each 80% interval (points); probability: C1."""
    rng = np.random.default_rng(seed)
    centre = np.asarray(log_points, float)
    spread = np.maximum(np.log1p(np.asarray(high80, float)) - centre, 1e-6) / Z80
    effort = np.maximum(np.expm1(rng.normal(centre, spread, (runs, len(centre)))), 0.0)
    committed = effort.sum(axis=1)
    spilled = (rng.random((runs, len(centre))) < np.asarray(probability, float)).sum(axis=1)
    result = {
        "stories": len(centre), "runs": runs,
        "committed_points": {"p10": _q(committed, 10), "p50": _q(committed, 50), "p90": _q(committed, 90)},
        "expected_stories_at_risk": round(float(np.sum(probability)), 2),
        "stories_at_risk_p90": int(np.percentile(spilled, 90)),
        "capacity_points": None, "overcommit_probability": None, "sprint_risk_level": None,
        "expected_overflow_points": None, "overflow_if_over_p50": None,
    }
    if capacity_mean is None or not np.isfinite(capacity_mean) or capacity_mean <= 0:
        return result  # without the team's capacity, over-commitment cannot be judged
    capacity_sd = (np.sqrt(capacity_variance) if capacity_variance is not None and np.isfinite(capacity_variance)
                   else DEFAULT_CAPACITY_SPREAD * capacity_mean)
    capacity = np.maximum(rng.normal(capacity_mean, capacity_sd, runs), 0.0)
    overflow = np.maximum(committed - capacity, 0.0)
    over = overflow > 0
    result.update({
        "capacity_points": {"p10": _q(capacity, 10), "p50": _q(capacity, 50), "p90": _q(capacity, 90)},
        "overcommit_probability": round(float(over.mean()), 4),
        "sprint_risk_level": level(float(over.mean())),
        "expected_overflow_points": round(float(overflow.mean()), 2),
        "overflow_if_over_p50": _q(overflow[over], 50) if over.any() else 0.0,
    })
    return result


def _q(values: np.ndarray, percentile: float) -> float:
    return round(float(np.percentile(values, percentile)), 2)
