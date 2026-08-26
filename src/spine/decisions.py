r"""Decision logic that turns a forecast or a probability into an action.

Holds: the newsvendor critical fractile and order quantity (parametric and
empirical), realised newsvendor cost, safety stock, and the cost-sensitive
classification threshold.

A forecast is not a decision. These functions are the bridge, and they are where
the asymmetry of real costs enters the analysis: the same forecast produces a
different order quantity depending on whether a stockout costs more than a unit
of spoilage, and the same probability produces a different approval depending on
whether a default costs more than a foregone margin.

Two of these functions share one idea. The newsvendor critical fractile

.. math:: CF = \frac{c_u}{c_u + c_o}

and the cost-sensitive classification threshold

.. math:: t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}

are the same ratio wearing different clothes: in both, the optimal action sits at
the point where the marginal expected cost of acting equals the marginal expected
cost of not acting. The classification threshold is a critical fractile over the
predicted probability rather than over the predicted demand.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from spine._checks import as_1d, as_binary, as_probabilities, same_length

__all__ = [
    "critical_fractile",
    "expected_cost_curve",
    "newsvendor_cost",
    "newsvendor_empirical",
    "newsvendor_parametric",
    "optimal_threshold",
    "safety_stock",
]


class _HasPPF(Protocol):
    """Anything exposing a quantile function, such as a frozen scipy distribution."""

    def ppf(self, q: float) -> Any:  # pragma: no cover - structural type only
        """Return the quantile at ``q``."""
        ...


def _check_costs(cu: float, co: float) -> None:
    """Reject non-positive underage or overage costs.

    Parameters
    ----------
    cu : float
        Underage cost.
    co : float
        Overage cost.

    Raises
    ------
    ValueError
        If either cost is not strictly positive.
    """
    if cu <= 0 or co <= 0:
        raise ValueError(f"cu and co must both be positive, got cu={cu}, co={co}")


def critical_fractile(cu: float, co: float) -> float:
    r"""Newsvendor critical fractile: the optimal service level.

    .. math:: CF = \frac{c_u}{c_u + c_o}

    The quantile of the demand distribution to order to. It is a *service level*,
    not a probability of profit: ordering to the 0.9 fractile means accepting a
    10% chance of a stockout because stockouts are nine times as expensive as
    leftovers.

    Parameters
    ----------
    cu : float
        Underage cost — the cost of being one unit short. Usually the lost
        margin, plus any goodwill cost of the stockout.
    co : float
        Overage cost — the cost of one unit left over. Usually the purchase cost
        less any salvage value, plus holding and spoilage.

    Returns
    -------
    float
        The optimal service level, strictly between 0 and 1.

    Raises
    ------
    ValueError
        If either cost is not strictly positive.

    Examples
    --------
    Losing 5 on a stockout and 2 on a leftover implies a 71.4% service level.

    >>> round(critical_fractile(5.0, 2.0), 6)
    0.714286
    """
    _check_costs(cu, co)
    return cu / (cu + co)


def newsvendor_parametric(distribution: _HasPPF, cu: float, co: float) -> float:
    r"""Optimal newsvendor order quantity for a parametric demand distribution.

    .. math:: Q^* = F^{-1}\!\left(\frac{c_u}{c_u + c_o}\right)

    Takes any object with a quantile function, so it is not restricted to normal
    demand. That matters: retail demand is usually right-skewed, and a normal
    approximation systematically under-orders in the tail — exactly the region a
    high critical fractile is asking about.

    Single period, no lead time, no lot sizing, no substitution. Those omissions
    are real limitations of the newsvendor model, not of this implementation.

    Parameters
    ----------
    distribution : object with a ``ppf`` method
        Frozen demand distribution, for example ``scipy.stats.norm(100, 20)``.
    cu : float
        Underage cost.
    co : float
        Overage cost.

    Returns
    -------
    float
        The order quantity that minimises expected cost.

    Raises
    ------
    TypeError
        If ``distribution`` has no ``ppf`` method.
    ValueError
        If either cost is not strictly positive.

    Examples
    --------
    Demand ~ Normal(100, 20) with cu=5 and co=2 gives a critical fractile of 5/7
    and an order of 100 + 20 * Phi^-1(5/7).

    >>> from scipy import stats
    >>> round(newsvendor_parametric(stats.norm(loc=100, scale=20), cu=5.0, co=2.0), 6)
    111.318976
    """
    if not hasattr(distribution, "ppf"):
        raise TypeError(
            f"distribution must expose a ppf (quantile) method, got "
            f"{type(distribution).__name__}; a frozen scipy distribution such as "
            f"stats.norm(loc=100, scale=20) works"
        )
    return float(distribution.ppf(critical_fractile(cu, co)))


def newsvendor_empirical(
    quantile_forecasts: ArrayLike,
    quantile_levels: ArrayLike,
    cu: float,
    co: float,
) -> float:
    r"""Optimal newsvendor order quantity from a grid of quantile forecasts.

    The entry point for a model that emits quantiles rather than a distribution.
    The order quantity is the forecast at the critical fractile, linearly
    interpolated between the two bracketing levels.

    Refuses to extrapolate. If the critical fractile falls outside the forecast
    grid — a 0.95 service level against forecasts that stop at 0.8 — clamping to
    the nearest available quantile would silently under-order at precisely the
    service level where the shortfall is most expensive. The error names the
    problem instead.

    Parameters
    ----------
    quantile_forecasts : array-like of shape (k,)
        Forecast demand at each level, ascending.
    quantile_levels : array-like of shape (k,)
        The quantile levels, strictly ascending, each in (0, 1).
    cu : float
        Underage cost.
    co : float
        Overage cost.

    Returns
    -------
    float
        Interpolated order quantity at the critical fractile.

    Raises
    ------
    ValueError
        If the inputs disagree in length, the levels are not ascending, or the
        critical fractile lies outside the grid.

    Examples
    --------
    A critical fractile of 5/7 sits between the 0.7 and 0.8 forecasts.

    >>> round(newsvendor_empirical([100.0, 110.0, 120.0], [0.5, 0.7, 0.8], 5.0, 2.0), 6)
    111.428571
    """
    forecasts = as_1d(quantile_forecasts, "quantile_forecasts")
    levels = as_1d(quantile_levels, "quantile_levels")
    same_length(("quantile_forecasts", forecasts), ("quantile_levels", levels))

    if not np.all(np.diff(levels) > 0):
        raise ValueError("quantile_levels must be strictly ascending")

    fractile = critical_fractile(cu, co)
    if fractile < levels[0] or fractile > levels[-1]:
        raise ValueError(
            f"the critical fractile {fractile:.6f} falls outside the quantile grid "
            f"[{levels[0]}, {levels[-1]}]; forecast the quantile the decision needs "
            f"rather than extrapolating from the ones you have"
        )
    return float(np.interp(fractile, levels, forecasts))


def newsvendor_cost(
    order_quantity: ArrayLike,
    realised_demand: ArrayLike,
    cu: float,
    co: float,
) -> NDArray[np.float64]:
    r"""Realised newsvendor cost, per period.

    .. math::

        C = c_u \max(D - Q,\,0) + c_o \max(Q - D,\,0)

    The number a business actually pays, as distinct from the forecast error.
    Two models with identical MASE can post very different realised costs when
    their errors point in different directions, which is the whole reason this
    function exists separately from the metrics module.

    Parameters
    ----------
    order_quantity : array-like or float
        Units ordered, per period.
    realised_demand : array-like or float
        Demand that actually occurred, per period.
    cu : float
        Underage cost per unit short.
    co : float
        Overage cost per unit left over.

    Returns
    -------
    numpy.ndarray
        Cost per period. Take the mean for a summary; the distribution is often
        more informative than its mean, since newsvendor costs are skewed.

    Examples
    --------
    Ordering 100 against demand of 120, 90 and 100:

    >>> newsvendor_cost([100.0, 100.0, 100.0], [120.0, 90.0, 100.0], 5.0, 2.0)
    array([100.,  20.,   0.])
    """
    _check_costs(cu, co)
    order = np.atleast_1d(np.asarray(order_quantity, dtype=np.float64))
    demand = np.atleast_1d(np.asarray(realised_demand, dtype=np.float64))
    if order.shape != demand.shape:
        raise ValueError(
            f"order_quantity and realised_demand must have the same shape, "
            f"got {order.shape} and {demand.shape}"
        )
    shortfall = np.maximum(demand - order, 0.0)
    excess = np.maximum(order - demand, 0.0)
    return cu * shortfall + co * excess


def safety_stock(demand_std: float, lead_time: float, service_level: float) -> float:
    r"""Safety stock for a target service level over a replenishment lead time.

    .. math:: SS = z_{\alpha}\,\sigma_D\,\sqrt{L}

    The square root is the whole content of the formula: demand variance
    accumulates linearly over independent periods, so the standard deviation
    grows with the square root of the lead time. Doubling the lead time raises
    the buffer by about 41%, not by 100%.

    Assumes demand is independent across periods and the lead time is
    deterministic. Both are usually false. Positive demand autocorrelation makes
    this an under-estimate, and lead time variability usually dominates demand
    variability in practice.

    Parameters
    ----------
    demand_std : float
        Standard deviation of demand per period.
    lead_time : float
        Replenishment lead time, in the same periods.
    service_level : float
        Target cycle service level, strictly between 0 and 1. Pair this with
        :func:`critical_fractile` when the service level should follow from
        costs rather than be chosen by hand.

    Returns
    -------
    float
        Buffer stock to hold above expected lead-time demand.

    Raises
    ------
    ValueError
        If any argument is outside its valid range.

    Examples
    --------
    Demand sd of 20 over a 4-period lead time at a 95% service level:

    >>> round(safety_stock(demand_std=20.0, lead_time=4.0, service_level=0.95), 6)
    65.794145
    """
    if demand_std < 0:
        raise ValueError(f"demand_std must not be negative, got {demand_std}")
    if lead_time <= 0:
        raise ValueError(f"lead_time must be positive, got {lead_time}")
    if not 0.0 < service_level < 1.0:
        raise ValueError(f"service_level must be strictly between 0 and 1, got {service_level}")
    return float(stats.norm.ppf(service_level) * demand_std * np.sqrt(lead_time))


def _validate_cost_matrix(cost_matrix: ArrayLike) -> NDArray[np.float64]:
    """Coerce and check a 2x2 cost matrix.

    Parameters
    ----------
    cost_matrix : array-like
        Costs indexed ``[true_label][predicted_label]``.

    Returns
    -------
    numpy.ndarray
        The matrix as float64.

    Raises
    ------
    ValueError
        If the matrix is not 2x2 or holds a negative cost.
    """
    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.shape != (2, 2):
        raise ValueError(f"cost_matrix must be 2x2, got shape {costs.shape}")
    if (costs < 0).any():
        raise ValueError("cost_matrix must not contain negative costs")
    return costs


def expected_cost_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    cost_matrix: ArrayLike,
) -> pd.DataFrame:
    """Expected cost per observation at every candidate threshold.

    The curve rather than the argmin, because its shape is the interesting part:
    a flat basin means the exact cutoff barely matters and the decision is robust
    to a mis-estimated cost; a sharp minimum means it matters a great deal and
    the cost assumptions need defending.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes. 1 is the event being predicted.
    y_prob : array-like of shape (n,)
        Predicted probabilities of the event.
    cost_matrix : array-like of shape (2, 2)
        Costs indexed ``[true_label][predicted_label]``, so ``cost_matrix[0][1]``
        is the false positive cost and ``cost_matrix[1][0]`` the false negative.

    Returns
    -------
    pandas.DataFrame
        Columns ``threshold`` and ``expected_cost``, sorted ascending by
        threshold. An observation is predicted positive when ``p >= threshold``.

    Examples
    --------
    >>> import numpy as np
    >>> costs = np.array([[0.0, 1.0], [5.0, 0.0]])
    >>> frame = expected_cost_curve([0, 1, 0, 1], [0.2, 0.3, 0.6, 0.7], costs)
    >>> frame["expected_cost"].round(6).tolist()
    [0.5, 0.25, 1.5, 1.25]
    """
    labels = as_binary(y_true, "y_true")
    probabilities = as_probabilities(y_prob, "y_prob")
    same_length(("y_true", labels), ("y_prob", probabilities))
    costs = _validate_cost_matrix(cost_matrix)

    # Evaluated in one pass rather than one pass per threshold. Predicting an
    # observation positive changes its cost by a fixed amount that depends only
    # on its true label, so the total cost at a threshold is the all-negative
    # cost plus the running sum of those changes over everything ranked above it.
    baseline = costs[labels, 0]
    delta = costs[labels, 1] - baseline

    order = np.argsort(-probabilities, kind="stable")
    running = np.concatenate([[0.0], np.cumsum(delta[order])])

    thresholds = np.unique(probabilities)
    n_predicted_positive = np.searchsorted(-probabilities[order], -thresholds, side="right")
    expected = (baseline.sum() + running[n_predicted_positive]) / len(labels)
    return pd.DataFrame({"threshold": thresholds, "expected_cost": expected})


def optimal_threshold(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    cost_matrix: ArrayLike,
) -> tuple[float, float]:
    r"""Cost-minimising classification threshold, chosen empirically.

    Searches the observed probabilities rather than assuming calibration. For a
    perfectly calibrated model the optimum is analytic,

    .. math:: t^* = \frac{C_{FP} - C_{TN}}{(C_{FP} - C_{TN}) + (C_{FN} - C_{TP})}

    and the empirical search converges to it. Where the two disagree, the
    disagreement measures miscalibration, and the analytic threshold is the one
    that is wrong — which is why a miscalibrated model cannot support a
    cost-based cutoff however well it ranks.

    Ties are broken towards the lower threshold.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes. 1 is the event being predicted.
    y_prob : array-like of shape (n,)
        Predicted probabilities of the event.
    cost_matrix : array-like of shape (2, 2)
        Costs indexed ``[true_label][predicted_label]``.

    Returns
    -------
    threshold : float
        Predict positive when ``p >= threshold``.
    expected_cost : float
        Mean cost per observation at that threshold, on this sample.

    Examples
    --------
    False negatives cost five times a false positive, so the cutoff sits low.

    >>> import numpy as np
    >>> costs = np.array([[0.0, 1.0], [5.0, 0.0]])
    >>> threshold, cost = optimal_threshold([0, 1, 0, 1], [0.2, 0.3, 0.6, 0.7], costs)
    >>> round(threshold, 6), round(cost, 6)
    (0.3, 0.25)
    """
    curve = expected_cost_curve(y_true, y_prob, cost_matrix)
    best = curve.loc[curve["expected_cost"].idxmin()]
    return float(best["threshold"]), float(best["expected_cost"])
