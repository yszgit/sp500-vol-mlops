"""Data validation: required columns present, no duplicates, sane values, no missing target."""

import numpy as np
import pandas as pd
import pytest

from volforecast import config
from volforecast.validation import DataValidationError, validate_panel, validate_training_frame


def test_clean_panel_passes(panel):
    assert validate_panel(panel) is panel


@pytest.mark.parametrize("column", config.REQUIRED_PANEL_COLUMNS)
def test_missing_required_column_is_rejected(panel, column):
    with pytest.raises(DataValidationError, match=column):
        validate_panel(panel.drop(columns=column))


def test_duplicate_rows_are_rejected(panel):
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_panel(pd.concat([panel, panel.head(100)], ignore_index=True))


def test_impossible_return_is_rejected(panel):
    bad = panel.copy()
    bad.loc[0, "return"] = -1.5  # a stock can't lose more than 100%
    with pytest.raises(DataValidationError, match="<= -100%"):
        validate_panel(bad)


def test_negative_volume_is_rejected(panel):
    bad = panel.copy()
    bad.loc[0, "volume"] = -10
    with pytest.raises(DataValidationError, match="negative volume"):
        validate_panel(bad)


def test_missing_target_is_rejected(splits):
    frame = splits["train"].copy()
    frame.loc[0, config.TARGET] = np.nan
    with pytest.raises(DataValidationError, match="missing target"):
        validate_training_frame(frame, config.ALL_FEATURES)


def test_model_frames_have_complete_features_and_target(splits):
    for frame in splits.values():
        validate_training_frame(frame, config.ALL_FEATURES)
