"""Time-ordered train / calibration / test split per project (ML guide 8.1 and Appendix B.2).

Random folds would let a model train on 2021 sprints and be tested on 2019 ones, using velocity and
spillover histories "from the future". Instead each project's sprints are ordered by start date: the oldest
~60% train the models, the next ~20% calibrate them (thresholds, C1, C2, early stopping), and the most recent
~20% are touched once, for the reported results. A story has one row (its first commitment), so no story
can appear on two sides.
"""

import numpy as np
import pandas as pd

TRAIN_SHARE = 0.6
CAL_SHARE = 0.2


def temporal_split(stories: pd.DataFrame, train: float = TRAIN_SHARE, cal: float = CAL_SHARE) -> pd.Series:
    """'train', 'cal' or 'test' per story. stories needs Project_ID, Sprint_ID and sprint_start."""
    split = pd.Series("test", index=stories.index, dtype=object)
    for _, group in stories.groupby("Project_ID"):
        order = group.sort_values(["sprint_start", "Sprint_ID"])["Sprint_ID"].drop_duplicates().to_numpy()
        a, b = int(len(order) * train), int(len(order) * (train + cal))
        split[group.index] = np.select(
            [group["Sprint_ID"].isin(order[:a]), group["Sprint_ID"].isin(order[a:b])], ["train", "cal"], "test")
    return split
