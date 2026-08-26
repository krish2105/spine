"""Tests for spine.decisions.

The newsvendor hand-check is cross-checked against ``statistics.NormalDist`` from
the standard library, which shares no code with the scipy the implementation uses.
That makes it an independent check rather than the same routine agreeing with
itself.
"""

from statistics import NormalDist

import numpy as np
import pytest
from scipy import stats

from spine.decisions import (
    critical_fractile,
    expected_cost_curve,
    newsvendor_cost,
    newsvendor_empirical,
    newsvendor_parametric,
    optimal_threshold,
    safety_stock,
)

# --------------------------------------------------------------- critical fractile


def test_critical_fractile_hand_computed():
    # CF = cu / (cu + co) = 5 / (5 + 2) = 5/7 = 0.714285714285...
    assert critical_fractile(5.0, 2.0) == pytest.approx(5 / 7, abs=1e-15)


def test_critical_fractile_is_one_half_when_costs_are_symmetric():
    assert critical_fractile(3.0, 3.0) == pytest.approx(0.5, abs=1e-15)


def test_critical_fractile_rises_as_understocking_gets_more_expensive():
    assert critical_fractile(9.0, 1.0) > critical_fractile(5.0, 2.0) > critical_fractile(1.0, 9.0)


def test_critical_fractile_rejects_non_positive_costs():
    for cu, co in [(0.0, 2.0), (-1.0, 2.0), (5.0, 0.0), (5.0, -2.0)]:
        with pytest.raises(ValueError, match="positive"):
            critical_fractile(cu, co)


# ------------------------------------------------------------ newsvendor, parametric


def test_newsvendor_parametric_matches_the_hand_computed_example():
    """Demand ~ Normal(100, 20), cu = 5, co = 2.

    By hand:
        CF  = cu / (cu + co) = 5 / (5 + 2) = 5/7 = 0.714285714285714...
        z   = Phi^-1(5/7)                  = 0.565948821932863...
        Q*  = mu + sigma * z = 100 + 20 * 0.565948821932863
            = 111.318976438657...

    The expected value is produced by statistics.NormalDist, which is standard
    library and independent of the scipy used by the implementation.
    """
    expected = NormalDist(100, 20).inv_cdf(5 / 7)
    assert expected == pytest.approx(111.318976438657, abs=1e-12)

    got = newsvendor_parametric(stats.norm(loc=100, scale=20), cu=5.0, co=2.0)
    assert got == pytest.approx(expected, abs=1e-6)
    assert round(got, 6) == 111.318976


def test_newsvendor_parametric_satisfies_the_optimality_condition():
    """The optimum is defined by F(Q*) = CF. Check the definition, not the formula."""
    demand = stats.norm(loc=100, scale=20)
    order = newsvendor_parametric(demand, cu=5.0, co=2.0)
    assert float(demand.cdf(order)) == pytest.approx(critical_fractile(5.0, 2.0), abs=1e-12)


def test_newsvendor_parametric_works_for_a_skewed_demand_distribution():
    # Nothing in the newsvendor argument assumes normality; it only needs a quantile.
    demand = stats.lognorm(s=0.5, scale=100)
    order = newsvendor_parametric(demand, cu=9.0, co=1.0)
    assert float(demand.cdf(order)) == pytest.approx(0.9, abs=1e-12)


def test_newsvendor_parametric_orders_more_when_stockouts_hurt_more():
    demand = stats.norm(loc=100, scale=20)
    assert newsvendor_parametric(demand, cu=9.0, co=1.0) > newsvendor_parametric(
        demand, cu=1.0, co=9.0
    )


def test_newsvendor_parametric_rejects_an_object_without_a_quantile_function():
    with pytest.raises(TypeError, match="ppf"):
        newsvendor_parametric(object(), cu=5.0, co=2.0)


# ------------------------------------------------------------- newsvendor, empirical


def test_newsvendor_empirical_interpolates_between_quantile_forecasts():
    # CF = 5/7 = 0.714285..., between the q=0.7 and q=0.8 forecasts.
    # Linear interpolation: 110 + (0.714285714 - 0.7) / (0.8 - 0.7) * (120 - 110)
    #                     = 110 + 0.14285714 * 10 = 111.4285714...
    got = newsvendor_empirical(
        quantile_forecasts=[100.0, 110.0, 120.0],
        quantile_levels=[0.5, 0.7, 0.8],
        cu=5.0,
        co=2.0,
    )
    assert got == pytest.approx(111.42857142857143, abs=1e-12)


def test_newsvendor_empirical_returns_the_exact_forecast_when_the_fractile_lands_on_one():
    # cu = co -> CF = 0.5 -> the median forecast, with no interpolation.
    got = newsvendor_empirical([100.0, 110.0, 120.0], [0.5, 0.7, 0.8], cu=3.0, co=3.0)
    assert got == pytest.approx(100.0, abs=1e-12)


def test_newsvendor_empirical_agrees_with_the_parametric_solution_on_a_dense_grid():
    """A dense quantile grid from a known distribution must recover the analytic Q*."""
    demand = stats.norm(loc=100, scale=20)
    levels = np.linspace(0.001, 0.999, 999)
    forecasts = demand.ppf(levels)
    empirical = newsvendor_empirical(forecasts, levels, cu=5.0, co=2.0)
    analytic = newsvendor_parametric(demand, cu=5.0, co=2.0)
    assert empirical == pytest.approx(analytic, abs=0.01)


def test_newsvendor_empirical_rejects_an_extrapolation_beyond_the_grid():
    # CF = 0.95 but the forecast grid stops at 0.8. Silently clamping to the top
    # forecast would understate the order and hide the problem.
    with pytest.raises(ValueError, match="outside the quantile grid"):
        newsvendor_empirical([100.0, 110.0], [0.5, 0.8], cu=19.0, co=1.0)


def test_newsvendor_empirical_rejects_unsorted_levels():
    with pytest.raises(ValueError, match="ascending"):
        newsvendor_empirical([100.0, 110.0], [0.8, 0.5], cu=5.0, co=2.0)


def test_newsvendor_empirical_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        newsvendor_empirical([100.0, 110.0, 120.0], [0.5, 0.8], cu=5.0, co=2.0)


# ------------------------------------------------------------------ realised cost


def test_newsvendor_cost_charges_understocking_and_overstocking_separately():
    # order 100 against demand 120 -> short by 20 -> 20 * cu(5) = 100
    # order 100 against demand  90 -> over by 10  -> 10 * co(2) =  20
    # order 100 against demand 100 -> exact                     =   0
    got = newsvendor_cost([100.0, 100.0, 100.0], [120.0, 90.0, 100.0], cu=5.0, co=2.0)
    assert got == pytest.approx(np.array([100.0, 20.0, 0.0]), abs=1e-12)


def test_newsvendor_cost_is_minimised_in_expectation_at_the_optimal_order():
    """Simulate: the analytic Q* must beat neighbouring order quantities on average."""
    rng = np.random.default_rng(20260827)
    demand = rng.normal(100, 20, size=200_000)
    optimal = newsvendor_parametric(stats.norm(loc=100, scale=20), cu=5.0, co=2.0)

    def mean_cost(order: float) -> float:
        return float(newsvendor_cost(np.full_like(demand, order), demand, 5.0, 2.0).mean())

    assert mean_cost(optimal) < mean_cost(optimal - 5.0)
    assert mean_cost(optimal) < mean_cost(optimal + 5.0)
    assert mean_cost(optimal) < mean_cost(100.0)  # and beats ordering the mean


def test_newsvendor_cost_accepts_scalars():
    assert newsvendor_cost(100.0, 120.0, cu=5.0, co=2.0) == pytest.approx(100.0, abs=1e-12)


# ------------------------------------------------------------------- safety stock


def test_safety_stock_hand_computed():
    # z(0.95) = 1.6448536269514722
    # ss = z * sigma * sqrt(L) = 1.6448536269514722 * 20 * sqrt(4)
    #    = 1.6448536269514722 * 40 = 65.79414507805889
    expected = NormalDist().inv_cdf(0.95) * 20 * 2
    assert expected == pytest.approx(65.794145078058, abs=1e-12)
    assert safety_stock(demand_std=20.0, lead_time=4.0, service_level=0.95) == pytest.approx(
        expected, abs=1e-9
    )


def test_safety_stock_scales_with_the_square_root_of_lead_time():
    one = safety_stock(20.0, 1.0, 0.95)
    four = safety_stock(20.0, 4.0, 0.95)
    assert four == pytest.approx(2 * one, abs=1e-9)


def test_safety_stock_is_zero_at_a_fifty_percent_service_level():
    assert safety_stock(20.0, 4.0, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_safety_stock_rejects_an_impossible_service_level():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="service_level"):
            safety_stock(20.0, 4.0, bad)


def test_safety_stock_rejects_non_positive_lead_time_or_negative_spread():
    with pytest.raises(ValueError, match="lead_time"):
        safety_stock(20.0, 0.0, 0.95)
    with pytest.raises(ValueError, match="demand_std"):
        safety_stock(-1.0, 4.0, 0.95)


# ------------------------------------------------------------ cost-sensitive cutoff

# cost_matrix[true][predicted]; the positive class is the event being predicted.
#   no cost for a correct call, 1 for a false positive, 5 for a false negative.
COSTS = np.array([[0.0, 1.0], [5.0, 0.0]])


def test_optimal_threshold_hand_computed():
    # y      = [0,   1,   0,   1  ]
    # y_prob = [0.2, 0.3, 0.6, 0.7]
    # Predict positive when p >= threshold. Mean cost at each candidate:
    #   t=0.2 -> [1,1,1,1] -> FP + TP + FP + TP = 2/4 = 0.50
    #   t=0.3 -> [0,1,1,1] -> TN + TP + FP + TP = 1/4 = 0.25   <- minimum
    #   t=0.6 -> [0,0,1,1] -> TN + FN + FP + TP = 6/4 = 1.50
    #   t=0.7 -> [0,0,0,1] -> TN + FN + TN + TP = 5/4 = 1.25
    threshold, cost = optimal_threshold([0, 1, 0, 1], [0.2, 0.3, 0.6, 0.7], COSTS)
    assert threshold == pytest.approx(0.3, abs=1e-12)
    assert cost == pytest.approx(0.25, abs=1e-12)


def test_optimal_threshold_approaches_the_analytic_optimum_when_probabilities_are_calibrated():
    """For a calibrated model the optimal cutoff is C_FP / (C_FP + C_FN).

    Here that is 1 / (1 + 5) = 0.1666..., and it does not depend on the sample.
    A model whose probabilities are wrong will not land here, which is why
    calibration matters more than ranking for a cost-based decision.
    """
    rng = np.random.default_rng(20260827)
    y_prob = rng.uniform(size=200_000)
    y_true = rng.binomial(1, y_prob)
    threshold, cost = optimal_threshold(y_true, y_prob, COSTS)
    # A loose tolerance, deliberately. The empirical argmin is noisy because the
    # cost basin is flat near the optimum; the assertion below is the sharp one.
    assert threshold == pytest.approx(1 / 6, abs=0.02)

    # What actually matters is that the cost is right, not that the threshold is.
    # Flatness is a relative claim, so assert it relatively.
    curve = expected_cost_curve(y_true, y_prob, COSTS)
    at_analytic = float(curve.iloc[(curve["threshold"] - 1 / 6).abs().idxmin()]["expected_cost"])
    assert (at_analytic - cost) / at_analytic < 0.005


def test_optimal_threshold_moves_down_as_false_negatives_get_more_expensive():
    rng = np.random.default_rng(3)
    y_prob = rng.uniform(size=20_000)
    y_true = rng.binomial(1, y_prob)
    cheap_fn, _ = optimal_threshold(y_true, y_prob, np.array([[0.0, 1.0], [2.0, 0.0]]))
    dear_fn, _ = optimal_threshold(y_true, y_prob, np.array([[0.0, 1.0], [20.0, 0.0]]))
    assert dear_fn < cheap_fn


def test_optimal_threshold_rejects_a_malformed_cost_matrix():
    with pytest.raises(ValueError, match="2x2"):
        optimal_threshold([0, 1], [0.2, 0.8], np.array([1.0, 2.0]))


def test_expected_cost_curve_contains_the_optimum_and_is_sorted():
    frame = expected_cost_curve([0, 1, 0, 1], [0.2, 0.3, 0.6, 0.7], COSTS)
    assert list(frame.columns) == ["threshold", "expected_cost"]
    assert frame["threshold"].is_monotonic_increasing
    best = frame.loc[frame["expected_cost"].idxmin()]
    assert best["threshold"] == pytest.approx(0.3, abs=1e-12)
    assert best["expected_cost"] == pytest.approx(0.25, abs=1e-12)


def test_expected_cost_curve_includes_the_decline_everything_endpoint():
    frame = expected_cost_curve([0, 1, 0, 1], [0.2, 0.3, 0.6, 0.7], COSTS)
    # Threshold at or below the smallest probability predicts positive for everyone:
    # two false positives out of four, at 1 each.
    assert frame["expected_cost"].iloc[0] == pytest.approx(0.5, abs=1e-12)
