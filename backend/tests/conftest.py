import numpy as np
import pytest


@pytest.fixture
def synthetic_landmarks():
    np.random.seed(42)
    arr = np.random.default_rng(0).uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    arr[:, 3] = 0.95
    return arr
