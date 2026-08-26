"""Tests for spine.metrics.

Every expectation here is either hand-computed in a comment above the assert, or
cross-checked against an independent implementation. No expectation is a number
copied out of a previous run of this code.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from spine.metrics import (
    brier_score,
    calibration_bins,
    expected_calibration_error,
    mase,
    mean_pinball_by_quantile,
    pinball_loss,
    pr_auc,
    qini_auc,
    qini_curve,
)

# --------------------------------------------------------------------------- MASE


def test_mase_hand_computed():
    # y_train = [1, 3, 2, 6], seasonality=1
    #   naive diffs: |3-1|, |2-3|, |6-2| = 2, 1, 4  -> scale = 7/3
    # y_true = [8, 4], y_pred = [7, 7]
    #   MAE = (|8-7| + |4-7|) / 2 = (1 + 3) / 2 = 2
    # MASE = 2 / (7/3) = 6/7 = 0.857142857...
    y_train = np.array([1.0, 3.0, 2.0, 6.0])
    got = mase([8.0, 4.0], [7.0, 7.0], y_train, seasonality=1)
    assert got == pytest.approx(6 / 7, abs=1e-12)


def test_mase_seasonal_denominator_uses_seasonality():
    # y_train = [1, 3, 2, 6], seasonality=2
    #   diffs at lag 2: |2-1|, |6-3| = 1, 3 -> scale = 2
    # y_true = [8, 4], y_pred = [7, 7] -> MAE = 2
    # MASE = 2 / 2 = 1.0
    y_train = np.array([1.0, 3.0, 2.0, 6.0])
    assert mase([8.0, 4.0], [7.0, 7.0], y_train, seasonality=2) == pytest.approx(1.0, abs=1e-12)


def test_mase_matches_utilsforecast_on_a_reference_series():
    """Cross-check against utilsforecast, an independent implementation.

    This is the check that the MASE denominator is the in-sample seasonal-naive
    MAE over the training window, not over the test window.
    """
    from utilsforecast.losses import mase as uf_mase

    rng = np.random.default_rng(20260827)
    seasonality = 7
    n_train, n_test = 120, 28
    t = np.arange(n_train + n_test)
    series = 100 + 10 * np.sin(2 * np.pi * t / seasonality) + rng.normal(0, 3, size=t.size)
    y_train, y_true = series[:n_train], series[n_train:]
    y_pred = y_true + rng.normal(0, 2, size=n_test)

    train_df = pd.DataFrame({"unique_id": "s1", "ds": pd.RangeIndex(n_train), "y": y_train})
    eval_df = pd.DataFrame(
        {
            "unique_id": "s1",
            "ds": pd.RangeIndex(n_train, n_train + n_test),
            "y": y_true,
            "model": y_pred,
        }
    )
    expected = uf_mase(eval_df, models=["model"], seasonality=seasonality, train_df=train_df)
    expected_value = float(expected["model"].iloc[0])

    assert mase(y_true, y_pred, y_train, seasonality=seasonality) == pytest.approx(
        expected_value, abs=1e-12
    )


def test_mase_rejects_seasonality_longer_than_training_window():
    with pytest.raises(ValueError, match="seasonality"):
        mase([1.0], [1.0], [1.0, 2.0], seasonality=5)


def test_mase_rejects_constant_training_window():
    # A flat training series makes the seasonal-naive MAE zero, so MASE is undefined.
    with pytest.raises(ValueError, match="zero"):
        mase([1.0, 2.0], [1.0, 1.0], [5.0, 5.0, 5.0], seasonality=1)


def test_mase_of_one_means_no_better_than_naive():
    # A naive forecast on a random walk scores close to 1.0 by construction.
    rng = np.random.default_rng(7)
    walk = np.cumsum(rng.normal(0, 1, size=400))
    y_train, y_true = walk[:300], walk[300:]
    naive = np.concatenate([[y_train[-1]], y_true[:-1]])
    assert mase(y_true, naive, y_train, seasonality=1) == pytest.approx(1.0, abs=0.15)


# ------------------------------------------------------------------- pinball loss


def test_pinball_loss_penalises_under_forecast_by_quantile():
    # q=0.9, y=10, yhat=8 -> under-forecast -> 0.9 * (10 - 8) = 1.8
    assert pinball_loss([10.0], [8.0], 0.9) == pytest.approx(1.8, abs=1e-12)


def test_pinball_loss_penalises_over_forecast_by_one_minus_quantile():
    # q=0.9, y=10, yhat=12 -> over-forecast -> 0.1 * (12 - 10) = 0.2
    assert pinball_loss([10.0], [12.0], 0.9) == pytest.approx(0.2, abs=1e-12)


def test_pinball_loss_at_median_is_half_absolute_error():
    y_true, y_pred = [10.0, 4.0], [7.0, 7.0]
    # 0.5 * mean(|10-7|, |4-7|) = 0.5 * 3 = 1.5
    assert pinball_loss(y_true, y_pred, 0.5) == pytest.approx(1.5, abs=1e-12)


def test_pinball_loss_rejects_quantile_outside_unit_interval():
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError, match="quantile"):
            pinball_loss([1.0], [1.0], bad)


def test_mean_pinball_by_quantile_returns_one_loss_per_level():
    y_true = np.array([10.0, 10.0])
    # columns are the q=0.1 and q=0.9 forecasts
    preds = np.array([[8.0, 12.0], [8.0, 12.0]])
    got = mean_pinball_by_quantile(y_true, preds, [0.1, 0.9])
    # q=0.1, yhat=8  -> under-forecast -> 0.1 * 2 = 0.2
    # q=0.9, yhat=12 -> over-forecast  -> 0.1 * 2 = 0.2
    assert got == pytest.approx(np.array([0.2, 0.2]), abs=1e-12)


def test_mean_pinball_by_quantile_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="quantile_levels"):
        mean_pinball_by_quantile([1.0, 2.0], np.zeros((2, 3)), [0.1, 0.9])


# -------------------------------------------------------------------------- PR-AUC


def test_pr_auc_hand_computed():
    # scores descending: 0.8(y=1), 0.7(y=0), 0.6(y=1), 0.5(y=0); 2 positives
    #   k=1: precision 1/1,   recall 1/2 -> dR 0.5 -> 0.500000
    #   k=2: precision 1/2,   recall 1/2 -> dR 0.0 -> 0
    #   k=3: precision 2/3,   recall 2/2 -> dR 0.5 -> 0.333333
    # average precision = 0.5 + 0.333333... = 0.8333333...
    y_true = [1, 0, 1, 0]
    y_score = [0.8, 0.7, 0.6, 0.5]
    assert pr_auc(y_true, y_score) == pytest.approx(5 / 6, abs=1e-12)


def test_pr_auc_matches_sklearn_average_precision():
    rng = np.random.default_rng(11)
    y_true = rng.binomial(1, 0.08, size=500)
    y_score = rng.uniform(size=500) + 0.4 * y_true
    assert pr_auc(y_true, y_score) == pytest.approx(
        average_precision_score(y_true, y_score), abs=1e-12
    )


def test_pr_auc_rejects_single_class():
    with pytest.raises(ValueError, match="both classes"):
        pr_auc([1, 1, 1], [0.2, 0.5, 0.9])


# ------------------------------------------------------------ Brier and calibration


def test_brier_score_hand_computed():
    # ((0.1-0)^2 + (0.2-0)^2 + (0.3-1)^2 + (0.9-1)^2) / 4
    #   = (0.01 + 0.04 + 0.49 + 0.01) / 4 = 0.55 / 4 = 0.1375
    assert brier_score([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9]) == pytest.approx(0.1375, abs=1e-12)


def test_expected_calibration_error_hand_computed():
    # n_bins=2, uniform edges at [0, 0.5, 1.0]
    #   bin 1: probs 0.1, 0.2, 0.3 -> mean conf 0.2, observed 1/3
    #          gap = |0.2 - 0.333333| = 0.133333, weight 3/4
    #   bin 2: prob 0.9            -> mean conf 0.9, observed 1
    #          gap = 0.1,                            weight 1/4
    # ECE = 0.75 * 0.1333333 + 0.25 * 0.1 = 0.1 + 0.025 = 0.125
    got = expected_calibration_error([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9], n_bins=2)
    assert got == pytest.approx(0.125, abs=1e-12)


def test_expected_calibration_error_is_zero_for_a_perfectly_calibrated_model():
    # Two groups: 100 cases at p=0.2 with exactly 20 positives, 100 at p=0.8 with 80.
    y_prob = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
    y_true = np.concatenate([np.repeat([1, 0], [20, 80]), np.repeat([1, 0], [80, 20])])
    assert expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(0.0, abs=1e-12)


def test_calibration_bins_quantile_strategy_balances_counts():
    rng = np.random.default_rng(3)
    y_prob = rng.uniform(size=200)
    y_true = rng.binomial(1, y_prob)
    frame = calibration_bins(y_true, y_prob, n_bins=4, strategy="quantile")
    assert list(frame.columns) == ["bin_lower", "bin_upper", "n", "mean_predicted", "observed_rate"]
    assert frame["n"].sum() == 200
    assert frame["n"].max() - frame["n"].min() <= 1


def test_calibration_bins_drops_empty_bins():
    # All mass in the top bin; the nine below it are empty and must not appear.
    frame = calibration_bins([1, 1, 0], [0.95, 0.96, 0.97], n_bins=10, strategy="uniform")
    assert len(frame) == 1
    assert frame["n"].iloc[0] == 3


def test_calibration_functions_reject_probabilities_outside_unit_interval():
    with pytest.raises(ValueError, match="probabilit"):
        brier_score([0, 1], [0.5, 1.4])
    with pytest.raises(ValueError, match="probabilit"):
        expected_calibration_error([0, 1], [-0.1, 0.5])


# ---------------------------------------------------------------------------- Qini

# Fixture, ordered by descending uplift score. T = treated, y = responded.
#   rank: 1   2   3   4   5   6   7   8
#   T:    1   0   1   0   1   0   1   0
#   y:    1   0   1   0   0   1   0   1
#
# Qini(t) = Y_T(t) - Y_C(t) * N_T(t) / N_C(t),  and 0 when N_C(t) == 0.
#   t=0: 0
#   t=1: N_T=1 Y_T=1 N_C=0        -> 1
#   t=2: N_T=1 Y_T=1 N_C=1 Y_C=0  -> 1
#   t=3: N_T=2 Y_T=2 N_C=1 Y_C=0  -> 2
#   t=4: N_T=2 Y_T=2 N_C=2 Y_C=0  -> 2
#   t=5: N_T=3 Y_T=2 N_C=2 Y_C=0  -> 2
#   t=6: N_T=3 Y_T=2 N_C=3 Y_C=1  -> 2 - 1 * (3/3) = 1
#   t=7: N_T=4 Y_T=2 N_C=3 Y_C=1  -> 2 - 1 * (4/3) = 0.666666...
#   t=8: N_T=4 Y_T=2 N_C=4 Y_C=2  -> 0
QINI_SCORE = np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20])
QINI_TREAT = np.array([1, 0, 1, 0, 1, 0, 1, 0])
QINI_Y = np.array([1, 0, 1, 0, 0, 1, 0, 1])
QINI_EXPECTED = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2 / 3, 0.0])


def test_qini_curve_hand_computed():
    fraction, gain = qini_curve(QINI_Y, QINI_TREAT, QINI_SCORE)
    assert fraction == pytest.approx(np.arange(9) / 8, abs=1e-12)
    assert gain == pytest.approx(QINI_EXPECTED, abs=1e-12)


def test_qini_auc_hand_computed():
    # Trapezoid over x in steps of 1/8 across QINI_EXPECTED:
    #   (0+1)/2 + (1+1)/2 + (1+2)/2 + (2+2)/2 + (2+2)/2 + (2+1)/2
    #     + (1+2/3)/2 + (2/3+0)/2
    #   = 0.5 + 1 + 1.5 + 2 + 2 + 1.5 + 0.8333333 + 0.3333333 = 9.6666666
    #   area = 9.6666666 / 8 = 1.2083333...
    # The random-targeting line runs from (0, 0) to (1, Qini(n)) = (1, 0),
    # so its area is 0 and the coefficient is the area under the curve.
    assert qini_auc(QINI_Y, QINI_TREAT, QINI_SCORE) == pytest.approx(29 / 24, abs=1e-12)


def test_qini_auc_is_near_zero_for_a_score_with_no_uplift_signal():
    """Treatment works, but the ranking does not, so the coefficient collapses.

    The coefficient is in units of incremental responders, so the tolerance has
    to be read against the total incremental response rather than against an
    absolute constant.
    """
    rng = np.random.default_rng(42)
    n = 4000
    treatment = rng.binomial(1, 0.5, size=n)
    y = rng.binomial(1, 0.2 + 0.1 * treatment)  # a real +0.1 average treatment effect

    _, gain = qini_curve(y, treatment, np.zeros(n))
    total_incremental = gain[-1]
    assert total_incremental > 150  # the effect is genuinely there

    # ...yet an uninformative ranking captures almost none of it.
    assert abs(qini_auc(y, treatment, np.zeros(n))) < 0.05 * total_incremental


def test_qini_ranks_a_perfect_score_above_a_random_one():
    rng = np.random.default_rng(5)
    n = 4000
    treatment = rng.binomial(1, 0.5, size=n)
    responder = rng.binomial(1, 0.5, size=n)  # only these people respond to treatment
    y = np.where(treatment == 1, responder, rng.binomial(1, 0.1, size=n))
    assert qini_auc(y, treatment, responder) > qini_auc(y, treatment, rng.uniform(size=n))


def test_qini_rejects_non_binary_treatment():
    with pytest.raises(ValueError, match="treatment"):
        qini_curve([0, 1], [0, 2], [0.1, 0.2])


def test_qini_rejects_a_single_treatment_arm():
    with pytest.raises(ValueError, match="both a treated and a control"):
        qini_curve([0, 1], [1, 1], [0.1, 0.2])


# ------------------------------------------------------------------ shared guards


def test_metrics_reject_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        brier_score([0, 1], [0.5])
    with pytest.raises(ValueError, match="same length"):
        pinball_loss([1.0, 2.0], [1.0], 0.5)


def test_metrics_reject_nan():
    with pytest.raises(ValueError, match="NaN"):
        brier_score([0, 1], [0.5, np.nan])
