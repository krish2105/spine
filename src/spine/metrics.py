"""Evaluation metrics for forecasting, classification and uplift.

Holds: MASE, pinball loss, PR-AUC, Brier score, expected calibration error,
calibration bins, Qini curve and Qini AUC.

Two deliberate omissions: MAPE (undefined at zero, asymmetric in its penalty for
over- versus under-forecasting, and not comparable across series of different
scale) and accuracy (uninformative under class imbalance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score

from spine._checks import as_1d, as_binary, as_probabilities, same_length

__all__ = [
    "brier_score",
    "calibration_bins",
    "expected_calibration_error",
    "mase",
    "mean_pinball_by_quantile",
    "pinball_loss",
    "pr_auc",
    "qini_auc",
    "qini_curve",
]


def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    seasonality: int = 1,
) -> float:
    r"""Mean Absolute Scaled Error.

    .. math::

        \mathrm{MASE} = \frac{\frac{1}{h}\sum_{t=1}^{h}|y_t - \hat{y}_t|}
                             {\frac{1}{n-m}\sum_{t=m+1}^{n}|y^{train}_t - y^{train}_{t-m}|}

    The denominator is the in-sample seasonal-naive mean absolute error, computed
    over the **training** window. ``y_train`` is a required positional argument
    for exactly that reason: scaling by the test window's own naive error is a
    leak, and this signature makes that version awkward to write by accident.

    MASE is used here in place of MAPE because MAPE is undefined wherever the
    actual is zero, penalises over- and under-forecasting asymmetrically, and
    cannot be compared across series of different scale. A MASE of 1.0 means the
    forecast is no better than a seasonal naive forecast, at any scale.

    Parameters
    ----------
    y_true : array-like of shape (h,)
        Observed values over the forecast horizon.
    y_pred : array-like of shape (h,)
        Forecast values over the same horizon.
    y_train : array-like of shape (n,)
        The training window the forecast was produced from. Must be in time order.
    seasonality : int, default 1
        Seasonal period ``m`` of the naive benchmark. Hourly 24, daily 7,
        weekly 52, monthly 12, quarterly 4, yearly 1.

    Returns
    -------
    float
        The scaled error. Below 1.0 beats seasonal naive; above 1.0 loses to it.

    Raises
    ------
    ValueError
        If ``seasonality`` is not shorter than ``y_train``, or if the training
        window's seasonal-naive error is zero, which leaves MASE undefined.

    Examples
    --------
    Training window [1, 3, 2, 6] has lag-1 differences 2, 1 and 4, so the scale is
    7/3. A forecast of [7, 7] against actuals of [8, 4] has an MAE of 2.

    >>> round(mase([8.0, 4.0], [7.0, 7.0], [1.0, 3.0, 2.0, 6.0], seasonality=1), 6)
    0.857143
    """
    y_true = as_1d(y_true, "y_true")
    y_pred = as_1d(y_pred, "y_pred")
    y_train = as_1d(y_train, "y_train")
    same_length(("y_true", y_true), ("y_pred", y_pred))

    if seasonality < 1:
        raise ValueError(f"seasonality must be at least 1, got {seasonality}")
    if seasonality >= len(y_train):
        raise ValueError(
            f"seasonality ({seasonality}) must be shorter than the training window "
            f"({len(y_train)}); there is no in-sample naive error to scale by"
        )

    scale = float(np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality])))
    if scale == 0.0:
        raise ValueError(
            "the seasonal-naive error over y_train is zero, so MASE is undefined; "
            "the training window is constant at this seasonality"
        )
    return float(np.mean(np.abs(y_true - y_pred))) / scale


def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, quantile: float) -> float:
    r"""Mean pinball (quantile) loss at a single quantile level.

    .. math::

        L_q(y, \hat{y}) = \begin{cases}
            q\,(y - \hat{y})     & y \geq \hat{y} \\
            (1 - q)(\hat{y} - y) & y < \hat{y}
        \end{cases}

    Asymmetric by construction: at ``q = 0.9`` an under-forecast costs nine times
    what an over-forecast of the same size costs. That asymmetry is the point —
    it is what makes the loss minimised by the qth quantile of the predictive
    distribution rather than by its mean.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed values.
    y_pred : array-like of shape (n,)
        Forecasts of the ``quantile`` quantile.
    quantile : float
        Quantile level, strictly between 0 and 1.

    Returns
    -------
    float
        Mean loss over the sample.

    Raises
    ------
    ValueError
        If ``quantile`` is not strictly inside (0, 1).

    Examples
    --------
    At q=0.9, under-forecasting 10 as 8 costs 0.9 x 2; over-forecasting it as 12
    costs only 0.1 x 2.

    >>> pinball_loss([10.0], [8.0], 0.9)
    1.8
    >>> round(pinball_loss([10.0], [12.0], 0.9), 12)
    0.2
    """
    y_true = as_1d(y_true, "y_true")
    y_pred = as_1d(y_pred, "y_pred")
    same_length(("y_true", y_true), ("y_pred", y_pred))
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be strictly between 0 and 1, got {quantile}")

    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def mean_pinball_by_quantile(
    y_true: ArrayLike,
    y_pred_quantiles: ArrayLike,
    quantile_levels: ArrayLike,
) -> NDArray[np.float64]:
    """Mean pinball loss at each of several quantile levels.

    The per-quantile breakdown, rather than a single averaged number, is what
    shows *where* a predictive distribution is miscalibrated — a model can be
    excellent at the median and useless in the tail the decision depends on.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed values.
    y_pred_quantiles : array-like of shape (n, k)
        Quantile forecasts, one column per level, in the order of
        ``quantile_levels``.
    quantile_levels : array-like of shape (k,)
        Quantile levels, each strictly between 0 and 1.

    Returns
    -------
    numpy.ndarray of shape (k,)
        Mean pinball loss per level.

    Raises
    ------
    ValueError
        If the number of columns does not match the number of levels.

    Examples
    --------
    >>> import numpy as np
    >>> preds = np.array([[8.0, 12.0], [8.0, 12.0]])
    >>> mean_pinball_by_quantile([10.0, 10.0], preds, [0.1, 0.9]).round(12)
    array([0.2, 0.2])
    """
    y_true = as_1d(y_true, "y_true")
    levels = as_1d(quantile_levels, "quantile_levels")
    predictions = np.asarray(y_pred_quantiles, dtype=np.float64)

    if predictions.ndim != 2:
        raise ValueError(f"y_pred_quantiles must be 2-D, got shape {predictions.shape}")
    if predictions.shape[0] != len(y_true):
        raise ValueError(
            f"y_pred_quantiles has {predictions.shape[0]} rows but y_true has {len(y_true)}"
        )
    if predictions.shape[1] != len(levels):
        raise ValueError(
            f"y_pred_quantiles has {predictions.shape[1]} columns but quantile_levels "
            f"has {len(levels)} entries"
        )

    return np.array(
        [pinball_loss(y_true, predictions[:, i], float(q)) for i, q in enumerate(levels)]
    )


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve, as average precision.

    Computed as average precision — the recall-weighted sum of precision at each
    threshold — rather than by trapezoidal interpolation of the PR curve.
    Trapezoidal interpolation between PR points is optimistically biased, because
    the curve between two operating points is not linear.

    Preferred over ROC-AUC under heavy class imbalance: ROC-AUC's false positive
    rate has a large denominator when negatives dominate, so a model can look
    strong while still returning mostly false positives at any usable threshold.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels, 0 or 1.
    y_score : array-like of shape (n,)
        Scores; higher means more likely positive. Need not be calibrated.

    Returns
    -------
    float
        Average precision. The no-skill baseline is the positive class prevalence,
        not 0.5.

    Raises
    ------
    ValueError
        If ``y_true`` does not contain both classes.

    Examples
    --------
    Scores in descending order 0.8(1), 0.7(0), 0.6(1), 0.5(0): precision is 1 at
    recall 0.5 and 2/3 at recall 1.0, giving 0.5 + 1/3.

    >>> round(pr_auc([1, 0, 1, 0], [0.8, 0.7, 0.6, 0.5]), 6)
    0.833333
    """
    labels = as_binary(y_true, "y_true")
    scores = as_1d(y_score, "y_score")
    same_length(("y_true", labels), ("y_score", scores))
    if len(np.unique(labels)) < 2:
        raise ValueError("y_true must contain both classes to define precision and recall")
    return float(average_precision_score(labels, scores))


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    r"""Brier score: mean squared error of a probability forecast.

    .. math::

        \mathrm{BS} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)^2

    A proper scoring rule, so it is minimised only by reporting the true
    probability. Unlike AUC it is sensitive to calibration, not just to ranking —
    which matters whenever a probability feeds a cost-based threshold, because a
    model that says 8% when the true rate is 15% produces the wrong cutoff no
    matter how well it ranks.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes, 0 or 1.
    y_prob : array-like of shape (n,)
        Predicted probabilities in [0, 1].

    Returns
    -------
    float
        Mean squared error. Lower is better; 0 is perfect.

    Examples
    --------
    >>> round(brier_score([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9]), 6)
    0.1375
    """
    labels = as_binary(y_true, "y_true")
    probabilities = as_probabilities(y_prob, "y_prob")
    same_length(("y_true", labels), ("y_prob", probabilities))
    return float(np.mean((probabilities - labels) ** 2))


def calibration_bins(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """Bin predicted probabilities and report the observed rate in each bin.

    This is the data behind a reliability diagram, returned as a table rather
    than plotted, so the caller owns the presentation.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes, 0 or 1.
    y_prob : array-like of shape (n,)
        Predicted probabilities in [0, 1].
    n_bins : int, default 10
        Number of bins.
    strategy : {"uniform", "quantile"}, default "uniform"
        ``"uniform"`` uses equal-width bins over [0, 1]; ``"quantile"`` uses
        equal-count bins. Quantile binning is the honest choice when predictions
        are concentrated in a narrow range, which is usual for rare events.

    Returns
    -------
    pandas.DataFrame
        One row per non-empty bin, with columns ``bin_lower``, ``bin_upper``,
        ``n``, ``mean_predicted`` and ``observed_rate``.

    Raises
    ------
    ValueError
        If ``n_bins`` is below 1 or ``strategy`` is not recognised.

    Examples
    --------
    >>> frame = calibration_bins([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9], n_bins=2)
    >>> frame[["n", "mean_predicted", "observed_rate"]].round(6).to_dict("records")
    [{'n': 3, 'mean_predicted': 0.2, 'observed_rate': 0.333333}, \
{'n': 1, 'mean_predicted': 0.9, 'observed_rate': 1.0}]
    """
    labels = as_binary(y_true, "y_true")
    probabilities = as_probabilities(y_prob, "y_prob")
    same_length(("y_true", labels), ("y_prob", probabilities))
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")

    index = np.searchsorted(edges[1:-1], probabilities, side="right")
    rows = []
    for bin_index in range(len(edges) - 1):
        mask = index == bin_index
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": float(edges[bin_index]),
                "bin_upper": float(edges[bin_index + 1]),
                "n": int(mask.sum()),
                "mean_predicted": float(probabilities[mask].mean()),
                "observed_rate": float(labels[mask].mean()),
            }
        )
    return pd.DataFrame(
        rows, columns=["bin_lower", "bin_upper", "n", "mean_predicted", "observed_rate"]
    )


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    r"""Expected Calibration Error.

    .. math::

        \mathrm{ECE} = \sum_{b=1}^{B}\frac{n_b}{n}\,
                       \bigl|\,\bar{p}_b - \bar{y}_b\,\bigr|

    The sample-weighted mean gap between mean predicted probability and observed
    rate across bins. Zero means the probabilities mean what they say.

    ECE is bin-dependent by construction: the number of bins and the binning
    strategy are part of the reported number, and should be stated with it.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes, 0 or 1.
    y_prob : array-like of shape (n,)
        Predicted probabilities in [0, 1].
    n_bins : int, default 10
        Number of bins.
    strategy : {"uniform", "quantile"}, default "uniform"
        Binning strategy, as in :func:`calibration_bins`.

    Returns
    -------
    float
        Weighted mean absolute calibration gap.

    Examples
    --------
    Two bins: three predictions averaging 0.2 against an observed rate of 1/3, and
    one prediction of 0.9 against an observed rate of 1.

    >>> round(expected_calibration_error([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9], n_bins=2), 6)
    0.125
    """
    frame = calibration_bins(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    weights = frame["n"] / frame["n"].sum()
    gaps = (frame["mean_predicted"] - frame["observed_rate"]).abs()
    return float((weights * gaps).sum())


def qini_curve(
    y: ArrayLike,
    treatment: ArrayLike,
    uplift_score: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Qini curve: incremental responders as a function of how many are targeted.

    .. math::

        Q(t) = Y^T_t - Y^C_t \frac{N^T_t}{N^C_t}

    where the population is sorted by descending uplift score and the superscripts
    denote the treated and control arms within the top ``t``. The control response
    is rescaled to the treated arm's size, so the curve reads in units of
    incremental responders attributable to treatment.

    Requires a randomised treatment assignment. On observational data the control
    rescaling has no causal interpretation.

    Ties in ``uplift_score`` are broken by input order, via a stable sort.

    Parameters
    ----------
    y : array-like of shape (n,)
        Binary outcome, 0 or 1.
    treatment : array-like of shape (n,)
        Binary treatment assignment, 1 for treated and 0 for control.
    uplift_score : array-like of shape (n,)
        Predicted uplift; higher means target sooner.

    Returns
    -------
    fraction_targeted : numpy.ndarray of shape (n + 1,)
        Fraction of the population targeted, from 0 to 1.
    incremental_response : numpy.ndarray of shape (n + 1,)
        Cumulative incremental responders at that targeting depth.

    Raises
    ------
    ValueError
        If ``treatment`` is not binary, or contains only one arm.

    Examples
    --------
    Alternating treated and control, with the two responders both treated and
    ranked first:

    >>> frac, gain = qini_curve([1, 0, 1, 0], [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])
    >>> gain.round(6)
    array([0., 1., 1., 2., 2.])
    """
    outcome = as_binary(y, "y")
    arm = as_binary(treatment, "treatment")
    score = as_1d(uplift_score, "uplift_score")
    same_length(("y", outcome), ("treatment", arm), ("uplift_score", score))
    if len(np.unique(arm)) < 2:
        raise ValueError("treatment must contain both a treated and a control arm")

    order = np.argsort(-score, kind="stable")
    outcome, arm = outcome[order], arm[order]

    n_treated = np.cumsum(arm)
    n_control = np.cumsum(1 - arm)
    responders_treated = np.cumsum(outcome * arm)
    responders_control = np.cumsum(outcome * (1 - arm))

    # Before any control unit has been reached the rescaling is undefined; the
    # incremental response is then just the treated response.
    ratio = np.divide(
        n_treated,
        n_control,
        out=np.zeros(len(outcome), dtype=np.float64),
        where=n_control > 0,
    )
    gain = responders_treated - responders_control * ratio

    fraction = np.arange(len(outcome) + 1, dtype=np.float64) / len(outcome)
    return fraction, np.concatenate([[0.0], gain])


def qini_auc(
    y: ArrayLike,
    treatment: ArrayLike,
    uplift_score: ArrayLike,
) -> float:
    """Qini coefficient: area between the Qini curve and random targeting.

    The random-targeting line runs from the origin to the curve's endpoint, which
    is the overall incremental response when everyone is targeted. Subtracting its
    area leaves only what the *ranking* contributed, so a score with no uplift
    signal scores 0 however large the average treatment effect happens to be.

    Not normalised by a perfect-model curve, so the value is in units of
    incremental responders per unit of population and is comparable only across
    models scored on the same sample. Distinct from AUUC, which integrates the
    unadjusted uplift curve and therefore rewards a large average treatment effect
    even when the ranking is uninformative.

    Parameters
    ----------
    y : array-like of shape (n,)
        Binary outcome, 0 or 1.
    treatment : array-like of shape (n,)
        Binary treatment assignment.
    uplift_score : array-like of shape (n,)
        Predicted uplift; higher means target sooner.

    Returns
    -------
    float
        Qini coefficient. Positive means the ranking beats random targeting.

    Examples
    --------
    The curve reaches 2 incremental responders, enclosing an area of 1.25 over
    the fraction axis. Random targeting to the same endpoint encloses 1.0, so the
    ranking is worth 0.25.

    >>> round(qini_auc([1, 0, 1, 0], [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]), 6)
    0.25
    """
    fraction, gain = qini_curve(y, treatment, uplift_score)
    return float(np.trapezoid(gain, fraction) - gain[-1] / 2.0)
