"""Temporal and group-aware splitters.

Every splitter in this module is temporal or group-aware. None of them shuffles.
A random split on temporal data lets a model see the future when predicting the
past, which inflates measured performance and does not replicate in production.

Holds: rolling-origin cross-validation (expanding and sliding window), a
group-aware temporal split for panel data, and a single-cutoff train/test split.

Two leaks these guard against:

Look-ahead through the boundary
    Handled by the geometry of the folds, and asserted as a property test:
    ``max(train) + gap < min(test)`` for every fold of every configuration.

Look-ahead through lagged features
    A model using a lag-``k`` feature reads ``k`` observations back. Training up
    to the instant the test window opens therefore still touches it. The ``gap``
    parameter embargoes those observations.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

__all__ = ["GroupTemporalSplit", "RollingOriginSplit", "temporal_train_test_split"]


class RollingOriginSplit:
    """Rolling-origin cross-validation over an ordered series.

    Folds are laid out from the end of the series backwards, so the final fold
    always tests on the most recent observations. The origin — the boundary
    between what is known and what is being predicted — moves forward by ``step``
    at each fold, which is what a forecaster actually does in production.

    Implements the scikit-learn splitter interface, so it can be passed as ``cv``
    to ``cross_val_score``, ``GridSearchCV`` and the rest.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    horizon : int, default 1
        Length of each test window, in observations.
    step : int, optional
        How far the origin advances between folds. Defaults to ``horizon``, which
        makes the test windows tile without overlapping.
    min_train_size : int, optional
        Smallest acceptable training window. Defaults to ``window_size`` for a
        sliding window and to 1 for an expanding one.
    window : {"expanding", "sliding"}, default "expanding"
        ``"expanding"`` trains on everything before the embargo; ``"sliding"``
        trains on a fixed-length window, which is the right choice when the
        data-generating process drifts and old data is actively misleading.
    window_size : int, optional
        Training window length. Required when ``window="sliding"``.
    gap : int, default 0
        Observations embargoed between the training window and the test window.
        Set this to the longest feature lag in the model.

    Raises
    ------
    ValueError
        If the configuration is not self-consistent, or if a series passed to
        :meth:`split` is too short to hold the requested folds.

    Examples
    --------
    Ten observations, three folds of two, tiling the end of the series:

    >>> import numpy as np
    >>> splitter = RollingOriginSplit(n_splits=3, horizon=2)
    >>> for train, test in splitter.split(np.arange(10)):
    ...     print(train.tolist(), "->", test.tolist())
    [0, 1, 2, 3] -> [4, 5]
    [0, 1, 2, 3, 4, 5] -> [6, 7]
    [0, 1, 2, 3, 4, 5, 6, 7] -> [8, 9]
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 1,
        step: int | None = None,
        min_train_size: int | None = None,
        window: str = "expanding",
        window_size: int | None = None,
        gap: int = 0,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {n_splits}")
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        if gap < 0:
            raise ValueError(f"gap must not be negative, got {gap}")
        if window not in {"expanding", "sliding"}:
            raise ValueError(f"window must be 'expanding' or 'sliding', got {window!r}")
        if window == "sliding" and window_size is None:
            raise ValueError("window_size is required when window='sliding'")
        if window_size is not None and window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}")

        step = horizon if step is None else step
        if step < 1:
            raise ValueError(f"step must be at least 1, got {step}")

        if min_train_size is None:
            min_train_size = window_size if window == "sliding" else 1
        if min_train_size < 1:
            raise ValueError(f"min_train_size must be at least 1, got {min_train_size}")

        self.n_splits = n_splits
        self.horizon = horizon
        self.step = step
        self.min_train_size = min_train_size
        self.window = window
        self.window_size = window_size
        self.gap = gap

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # noqa: N803
        """Return the number of folds, for the scikit-learn splitter interface.

        Parameters
        ----------
        X, y, groups : ignored
            Present for interface compatibility.

        Returns
        -------
        int
            The configured number of folds.
        """
        return self.n_splits

    def split(
        self,
        X: ArrayLike,  # noqa: N803
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        """Generate integer index pairs for each fold.

        Parameters
        ----------
        X : array-like or DataFrame
            Only its length is used. Rows must already be in time order.
        y, groups : ignored
            Present for interface compatibility.

        Yields
        ------
        train_index : numpy.ndarray
            Positions of the training rows, ascending and contiguous.
        test_index : numpy.ndarray
            Positions of the test rows, ascending and contiguous.

        Raises
        ------
        ValueError
            If the series is too short for the configured folds.
        """
        n_samples = len(X)  # type: ignore[arg-type]
        required = self.min_train_size + self.gap + self.horizon + (self.n_splits - 1) * self.step
        if n_samples < required:
            raise ValueError(
                f"a {self.n_splits}-fold split with horizon={self.horizon}, "
                f"step={self.step}, gap={self.gap} and min_train_size="
                f"{self.min_train_size} needs at least {required} observations, "
                f"got {n_samples}"
            )

        for fold in range(self.n_splits):
            test_end = n_samples - (self.n_splits - 1 - fold) * self.step
            test_start = test_end - self.horizon
            train_end = test_start - self.gap

            if self.window == "sliding":
                train_start = max(0, train_end - int(self.window_size))  # type: ignore[arg-type]
            else:
                train_start = 0

            yield (
                np.arange(train_start, train_end, dtype=int),
                np.arange(test_start, test_end, dtype=int),
            )


class GroupTemporalSplit:
    """Rolling-origin cross-validation over unique time keys, for panel data.

    A panel has many rows per timestamp. Splitting it by row position cuts a
    timestamp in half, putting some of a day's observations in training and the
    rest in test — a leak that is invisible in the fold sizes and easy to ship.
    This splitter cuts on the unique sorted time keys instead, so every row
    sharing a timestamp lands on the same side.

    Optionally also removes entities that appear on both sides. Scoring a model
    on a customer it was trained on measures memorisation, not generalisation;
    entities are dropped from the *test* side, never from training, so no
    training signal is discarded.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    horizon : int, default 1
        Length of each test window, in **distinct time keys** rather than rows.
    step : int, optional
        Time keys the origin advances between folds. Defaults to ``horizon``.
    min_train_size : int, optional
        Smallest acceptable training window, in time keys.
    window : {"expanding", "sliding"}, default "expanding"
        As in :class:`RollingOriginSplit`.
    window_size : int, optional
        Training window length in time keys. Required when ``window="sliding"``.
    gap : int, default 0
        Time keys embargoed between training and test.

    Examples
    --------
    Two rows per day across four days, testing on the last day:

    >>> import numpy as np, pandas as pd
    >>> days = np.repeat(pd.date_range("2026-01-01", periods=4, freq="D"), 2)
    >>> train, test = next(iter(GroupTemporalSplit(n_splits=1, horizon=1).split(days)))
    >>> train.tolist(), test.tolist()
    ([0, 1, 2, 3, 4, 5], [6, 7])
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 1,
        step: int | None = None,
        min_train_size: int | None = None,
        window: str = "expanding",
        window_size: int | None = None,
        gap: int = 0,
    ) -> None:
        self._inner = RollingOriginSplit(
            n_splits=n_splits,
            horizon=horizon,
            step=step,
            min_train_size=min_train_size,
            window=window,
            window_size=window_size,
            gap=gap,
        )
        self.n_splits = n_splits

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # noqa: N803
        """Return the number of folds.

        Parameters
        ----------
        X, y, groups : ignored
            Present for interface compatibility.

        Returns
        -------
        int
            The configured number of folds.
        """
        return self.n_splits

    def split(
        self,
        time_keys: ArrayLike,
        entities: ArrayLike | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        """Generate index pairs cut on time-key boundaries.

        Parameters
        ----------
        time_keys : array-like of shape (n_rows,)
            The time label of each row. Need not be sorted, and repeats are the
            normal case. Any orderable dtype works.
        entities : array-like of shape (n_rows,), optional
            Entity label of each row. When given, entities present in the
            training side are removed from the test side.

        Yields
        ------
        train_index : numpy.ndarray
            Row positions for training, ascending.
        test_index : numpy.ndarray
            Row positions for testing, ascending.

        Raises
        ------
        ValueError
            If ``entities`` has a different length from ``time_keys``, or the
            panel holds too few distinct time keys for the configured folds.
        """
        keys = np.asarray(time_keys)
        if entities is not None:
            entities = np.asarray(entities)
            if len(entities) != len(keys):
                raise ValueError(
                    f"time_keys and entities must have the same length: "
                    f"{len(keys)} and {len(entities)}"
                )

        unique_keys = np.unique(keys)
        for key_train, key_test in self._inner.split(unique_keys):
            train = np.flatnonzero(np.isin(keys, unique_keys[key_train]))
            test = np.flatnonzero(np.isin(keys, unique_keys[key_test]))

            if entities is not None:
                seen = set(entities[train].tolist())
                test = test[[entity not in seen for entity in entities[test]]]

            yield train, test


def temporal_train_test_split(
    frame: pd.DataFrame,
    time_col: str,
    cutoff: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame at a single point in time.

    Rows strictly before ``cutoff`` become training; ``cutoff`` itself and
    everything after become test. The frame is sorted by ``time_col`` first, so
    an unsorted input cannot silently produce a scrambled split.

    Both sides must be non-empty. A split that yields nothing on one side is
    almost always a cutoff typed in the wrong units or the wrong year, and
    failing loudly is cheaper than discovering it three notebooks later.

    Parameters
    ----------
    frame : pandas.DataFrame
        The data to split.
    time_col : str
        Name of the column holding the time index.
    cutoff : scalar
        First moment of the test period. Compared directly against ``time_col``,
        so a string is fine for a datetime column.

    Returns
    -------
    train : pandas.DataFrame
        Rows before the cutoff, sorted by time.
    test : pandas.DataFrame
        Rows from the cutoff onwards, sorted by time.

    Raises
    ------
    KeyError
        If ``time_col`` is not a column of ``frame``.
    ValueError
        If either side of the split is empty.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame(
    ...     {"date": pd.date_range("2026-01-01", periods=4), "value": [1, 2, 3, 4]}
    ... )
    >>> train, test = temporal_train_test_split(frame, "date", "2026-01-03")
    >>> train["value"].tolist(), test["value"].tolist()
    ([1, 2], [3, 4])
    """
    if time_col not in frame.columns:
        raise KeyError(f"{time_col!r} is not a column of the frame; found {list(frame.columns)}")

    ordered = frame.sort_values(time_col, kind="stable")
    boundary = pd.Series([cutoff]).astype(ordered[time_col].dtype).iloc[0]
    is_train = ordered[time_col] < boundary

    train, test = ordered[is_train], ordered[~is_train]
    if train.empty or test.empty:
        raise ValueError(
            f"cutoff {cutoff!r} leaves one side empty "
            f"({len(train)} training rows, {len(test)} test rows); "
            f"the time column spans {ordered[time_col].min()} to {ordered[time_col].max()}"
        )
    return train, test
