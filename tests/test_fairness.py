"""Tests for spine.fairness.

Expectations are hand-computed from a four-row-per-group fixture, small enough
that every rate can be checked by counting. That is the point of implementing
these directly rather than wrapping a library: a fairness number that goes into a
regulatory report should be traceable to arithmetic someone can redo on paper.
"""

import numpy as np
import pandas as pd
import pytest

from spine.fairness import calibration_by_group, disparity_ratio, group_metrics

# Group "a": y_true [1,1,0,0], y_pred [1,0,1,0]
#   selected 2/4 = 0.5 | TPR 1/2 = 0.5 | FPR 1/2 = 0.5 | PPV 1/2 = 0.5
# Group "b": y_true [1,1,1,0], y_pred [1,1,0,0]
#   selected 2/4 = 0.5 | TPR 2/3       | FPR 0/1 = 0.0 | PPV 2/2 = 1.0
Y_TRUE = [1, 1, 0, 0, 1, 1, 1, 0]
Y_PRED = [1, 0, 1, 0, 1, 1, 0, 0]
GROUPS = ["a", "a", "a", "a", "b", "b", "b", "b"]


def test_group_metrics_hand_computed():
    frame = group_metrics(Y_TRUE, Y_PRED, GROUPS)
    assert list(frame.columns) == ["n", "selection_rate", "tpr", "fpr", "ppv"]
    assert frame.index.tolist() == ["a", "b"]

    assert frame.loc["a"].to_dict() == pytest.approx(
        {"n": 4, "selection_rate": 0.5, "tpr": 0.5, "fpr": 0.5, "ppv": 0.5}
    )
    assert frame.loc["b"].to_dict() == pytest.approx(
        {"n": 4, "selection_rate": 0.5, "tpr": 2 / 3, "fpr": 0.0, "ppv": 1.0}
    )


def test_group_metrics_returns_nan_where_a_rate_is_undefined():
    """Group "c" has no positive outcomes, so TPR has a zero denominator.

    NaN, not zero. Zero would read as "this group scores badly" when the truth is
    "this sample cannot answer the question for this group".
    """
    frame = group_metrics([0, 0, 1, 0], [1, 0, 1, 0], ["c", "c", "d", "d"])
    assert np.isnan(frame.loc["c", "tpr"])
    assert frame.loc["c", "fpr"] == pytest.approx(0.5)


def test_group_metrics_returns_nan_for_ppv_when_nothing_is_selected():
    frame = group_metrics([1, 0, 1, 0], [0, 0, 1, 0], ["d", "d", "e", "e"])
    assert np.isnan(frame.loc["d", "ppv"])
    assert frame.loc["d", "selection_rate"] == pytest.approx(0.0)


def test_group_metrics_index_holds_plain_python_labels():
    """The index ends up in a report a human reads; np.str_('a') is noise."""
    frame = group_metrics([1, 0, 1, 0], [1, 0, 1, 0], np.array(["a", "a", "b", "b"]))
    assert [type(label) for label in frame.index] == [str, str]
    assert frame["n"].dtype.kind == "i"


def test_group_metrics_sorts_groups_so_reports_are_stable():
    frame = group_metrics([1, 0, 1], [1, 0, 1], ["z", "a", "m"])
    assert frame.index.tolist() == ["a", "m", "z"]


def test_group_metrics_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        group_metrics([1, 0], [1, 0], ["a"])


def test_group_metrics_rejects_a_single_group():
    with pytest.raises(ValueError, match="at least two groups"):
        group_metrics([1, 0], [1, 0], ["a", "a"])


# ------------------------------------------------------------------ disparity ratio


def test_disparity_ratio_hand_computed():
    frame = group_metrics(Y_TRUE, Y_PRED, GROUPS)
    # selection rate: min 0.5 / max 0.5 = 1.0  -> parity
    assert disparity_ratio(frame, "selection_rate") == pytest.approx(1.0, abs=1e-12)
    # TPR:            min 0.5 / max 2/3 = 0.75 -> below the four-fifths threshold
    assert disparity_ratio(frame, "tpr") == pytest.approx(0.75, abs=1e-12)
    # PPV:            min 0.5 / max 1.0 = 0.5
    assert disparity_ratio(frame, "ppv") == pytest.approx(0.5, abs=1e-12)


def test_disparity_ratio_against_a_named_reference_group():
    frame = group_metrics(Y_TRUE, Y_PRED, GROUPS)
    # b's TPR relative to a's: (2/3) / 0.5 = 1.333..., a ratio above 1 because the
    # reference group is the worse-off one.
    assert disparity_ratio(frame, "tpr", reference="a") == pytest.approx(4 / 3, abs=1e-12)


def test_disparity_ratio_ignores_undefined_rates():
    frame = pd.DataFrame({"tpr": [0.8, np.nan, 0.4]}, index=["a", "b", "c"])
    assert disparity_ratio(frame, "tpr") == pytest.approx(0.5, abs=1e-12)


def test_disparity_ratio_rejects_a_missing_column():
    frame = group_metrics(Y_TRUE, Y_PRED, GROUPS)
    with pytest.raises(KeyError, match="accuracy"):
        disparity_ratio(frame, "accuracy")


def test_disparity_ratio_rejects_a_missing_reference_group():
    frame = group_metrics(Y_TRUE, Y_PRED, GROUPS)
    with pytest.raises(KeyError, match="c"):
        disparity_ratio(frame, "tpr", reference="c")


def test_disparity_ratio_is_undefined_when_every_rate_is_undefined():
    frame = pd.DataFrame({"tpr": [np.nan, np.nan]}, index=["a", "b"])
    assert np.isnan(disparity_ratio(frame, "tpr"))


# ------------------------------------------------------------- calibration by group


def test_calibration_by_group_hand_computed():
    # Group "a": both predicted 0.2, neither happened -> the model over-predicts by 0.2
    # Group "b": both predicted 0.8, both happened    -> under-predicts by 0.2
    frame = calibration_by_group(
        y_true=[0, 0, 1, 1],
        y_prob=[0.2, 0.2, 0.8, 0.8],
        sensitive=["a", "a", "b", "b"],
    )
    assert list(frame.columns) == ["n", "mean_predicted", "observed_rate", "calibration_gap"]
    assert frame.loc["a"].to_dict() == pytest.approx(
        {"n": 2, "mean_predicted": 0.2, "observed_rate": 0.0, "calibration_gap": 0.2}
    )
    assert frame.loc["b"].to_dict() == pytest.approx(
        {"n": 2, "mean_predicted": 0.8, "observed_rate": 1.0, "calibration_gap": -0.2}
    )


def test_calibration_by_group_gap_is_zero_when_each_group_is_calibrated():
    y_prob = np.concatenate([np.full(100, 0.3), np.full(100, 0.7)])
    y_true = np.concatenate([np.repeat([1, 0], [30, 70]), np.repeat([1, 0], [70, 30])])
    sensitive = np.repeat(["a", "b"], 100)
    frame = calibration_by_group(y_true, y_prob, sensitive)
    assert frame["calibration_gap"].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_calibration_by_group_rejects_probabilities_outside_the_unit_interval():
    with pytest.raises(ValueError, match="probabilit"):
        calibration_by_group([0, 1], [0.5, 1.2], ["a", "b"])
