"""Group fairness metrics and disparity ratios.

Holds: group-wise selection rate, TPR, FPR and PPV; the disparity ratio
(four-fifths rule); and calibration by group.

Implemented directly in numpy/pandas rather than wrapping a fairness library, so
that every number on a fairness report can be traced to arithmetic that fits on
one screen. Nothing here is novel; the value is that it is auditable.

Two things this module will not do for you.

It will not tell you which metric to optimise.
    Under unequal base rates, calibration by group and equalised odds are
    mathematically incompatible — you cannot have both, and a tool that hid that
    would be lying. Choose, and report the metric you did not choose.

It will not turn a proxy into a protected attribute.
    Where the real attribute is unavailable and a proxy stands in, every result
    derived from it is a methodological demonstration. :mod:`spine.cards` refuses
    to render a fairness section that does not say which it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from spine._checks import as_binary, as_probabilities, same_length

__all__ = ["calibration_by_group", "disparity_ratio", "group_metrics"]


def _label(level: object) -> object:
    """Convert a numpy scalar group label back to a plain Python object.

    Group labels end up as the index of a report that a human reads. A numpy
    scalar renders as ``np.str_('a')`` rather than ``'a'``, which is noise.

    Parameters
    ----------
    level : object
        A group label, possibly a numpy scalar.

    Returns
    -------
    object
        The equivalent Python object.
    """
    return level.item() if hasattr(level, "item") else level


def _rate(numerator: int, denominator: int) -> float:
    """Divide, returning NaN rather than raising when the denominator is zero.

    Parameters
    ----------
    numerator : int
        Count of the events of interest.
    denominator : int
        Count of the opportunities for that event.

    Returns
    -------
    float
        The rate, or NaN when it is undefined.
    """
    return float("nan") if denominator == 0 else numerator / denominator


def group_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive: ArrayLike,
) -> pd.DataFrame:
    """Confusion-derived rates for each level of a sensitive attribute.

    Rates that are undefined for a group — TPR where a group has no positive
    outcomes, PPV where nothing was selected — come back as NaN rather than as
    zero. Zero would read as "this group scores badly", when the truth is "this
    sample cannot answer the question for this group", and the two call for very
    different responses.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes, 0 or 1.
    y_pred : array-like of shape (n,)
        Binary decisions, 0 or 1. The positive class is the selected one — the
        approval, the offer, the flag — whichever is the action being audited.
    sensitive : array-like of shape (n,)
        Group label per observation. Any hashable dtype.

    Returns
    -------
    pandas.DataFrame
        Indexed by group, sorted by group label so that reports are stable
        across runs. Columns: ``n``, ``selection_rate``, ``tpr``, ``fpr``,
        ``ppv``.

    Raises
    ------
    ValueError
        If the inputs disagree in length, or fewer than two groups are present.

    Examples
    --------
    >>> frame = group_metrics([1, 1, 0, 0], [1, 0, 1, 0], ["a", "a", "b", "b"])
    >>> frame[["n", "selection_rate"]].to_dict("index")
    {'a': {'n': 2, 'selection_rate': 0.5}, 'b': {'n': 2, 'selection_rate': 0.5}}
    """
    truth = as_binary(y_true, "y_true")
    predicted = as_binary(y_pred, "y_pred")
    groups = np.asarray(sensitive)
    same_length(("y_true", truth), ("y_pred", predicted), ("sensitive", groups))

    levels = np.unique(groups)
    if len(levels) < 2:
        raise ValueError(f"a disparity needs at least two groups to compare, found {len(levels)}")

    rows = {}
    for level in levels:
        mask = groups == level
        actual, decision = truth[mask], predicted[mask]
        positives = int(actual.sum())
        negatives = int((1 - actual).sum())
        selected = int(decision.sum())
        rows[_label(level)] = {
            "n": int(mask.sum()),
            "selection_rate": _rate(selected, int(mask.sum())),
            "tpr": _rate(int(((actual == 1) & (decision == 1)).sum()), positives),
            "fpr": _rate(int(((actual == 0) & (decision == 1)).sum()), negatives),
            "ppv": _rate(int(((actual == 1) & (decision == 1)).sum()), selected),
        }

    frame = pd.DataFrame.from_dict(rows, orient="index")[
        ["n", "selection_rate", "tpr", "fpr", "ppv"]
    ]
    return frame.astype({"n": int})


def disparity_ratio(
    frame: pd.DataFrame,
    column: str,
    reference: object | None = None,
) -> float:
    """Ratio between the least and most favoured group on one metric.

    With no reference group this is the min-over-max ratio behind the
    four-fifths rule: a value below 0.8 is the conventional trigger for closer
    scrutiny. That threshold is a rule of thumb from US employment guidance, not
    a legal standard anywhere this library is used, and it is a screening
    heuristic rather than a verdict.

    Groups whose metric is undefined are excluded. A ratio computed over a single
    remaining group is meaningless, so it returns NaN.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`group_metrics`, or any frame indexed by group.
    column : str
        Metric to compare.
    reference : hashable, optional
        Group to compare every other group against. When given, the ratio is
        ``min(other groups) / reference``, which can exceed 1 if the reference
        group is the worse-off one. When omitted, ``min / max``.

    Returns
    -------
    float
        The ratio, or NaN when it cannot be computed.

    Raises
    ------
    KeyError
        If ``column`` is not in the frame, or ``reference`` is not in the index.

    Examples
    --------
    >>> frame = group_metrics(
    ...     [1, 1, 0, 0, 1, 1, 1, 0], [1, 0, 1, 0, 1, 1, 0, 0], list("aaaabbbb")
    ... )
    >>> round(disparity_ratio(frame, "tpr"), 6)
    0.75
    """
    if column not in frame.columns:
        raise KeyError(f"{column!r} is not a column of the frame; found {list(frame.columns)}")

    values = frame[column].dropna()
    if values.empty:
        return float("nan")

    if reference is not None:
        if reference not in frame.index:
            raise KeyError(f"{reference!r} is not a group in the frame; found {list(frame.index)}")
        baseline = frame.loc[reference, column]
        others = values.drop(index=reference, errors="ignore")
        if others.empty or baseline == 0 or pd.isna(baseline):
            return float("nan")
        return float(others.min() / baseline)

    if len(values) < 2 or values.max() == 0:
        return float("nan")
    return float(values.min() / values.max())


def calibration_by_group(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    sensitive: ArrayLike,
) -> pd.DataFrame:
    """Mean predicted probability against observed rate, per group.

    The fairness metric that matters most for a pricing or credit decision. A
    group whose predicted probabilities run consistently above its observed rate
    is being systematically over-charged for risk it does not carry, and no
    threshold choice downstream can repair that — the input to the decision is
    simply wrong for those people.

    Reported as a signed gap, not an absolute one, because the direction is the
    finding: over-prediction and under-prediction harm different people.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes, 0 or 1.
    y_prob : array-like of shape (n,)
        Predicted probabilities in [0, 1].
    sensitive : array-like of shape (n,)
        Group label per observation.

    Returns
    -------
    pandas.DataFrame
        Indexed by group. Columns: ``n``, ``mean_predicted``, ``observed_rate``,
        ``calibration_gap`` (predicted minus observed; positive means the model
        over-predicts risk for that group).

    Examples
    --------
    >>> frame = calibration_by_group([0, 0, 1, 1], [0.2, 0.2, 0.8, 0.8], list("aabb"))
    >>> frame["calibration_gap"].round(6).to_dict()
    {'a': 0.2, 'b': -0.2}
    """
    truth = as_binary(y_true, "y_true")
    probabilities = as_probabilities(y_prob, "y_prob")
    groups = np.asarray(sensitive)
    same_length(("y_true", truth), ("y_prob", probabilities), ("sensitive", groups))

    rows = {}
    for level in np.unique(groups):
        mask = groups == level
        predicted = float(probabilities[mask].mean())
        observed = float(truth[mask].mean())
        rows[_label(level)] = {
            "n": int(mask.sum()),
            "mean_predicted": predicted,
            "observed_rate": observed,
            "calibration_gap": predicted - observed,
        }
    frame = pd.DataFrame.from_dict(rows, orient="index")[
        ["n", "mean_predicted", "observed_rate", "calibration_gap"]
    ]
    return frame.astype({"n": int})
