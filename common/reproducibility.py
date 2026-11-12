"""Seed standard Python and NumPy code; model libraries need their own seed too."""

import random

import numpy as np

from . import SEED


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
