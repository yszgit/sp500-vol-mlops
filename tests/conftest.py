import pytest

from volforecast.features import build_features, make_splits
from volforecast.synthetic import make_synthetic_panel


@pytest.fixture(scope="session")
def panel():
    return make_synthetic_panel(n_stocks=40, seed=42)


@pytest.fixture(scope="session")
def splits(panel):
    return make_splits(build_features(panel))
