"""Tests for spine.splitting.

The central test in this file is the property test: for any generated series and
any valid configuration, no training index may reach past the embargo into its
own test window. That is the guarantee the whole library rests on, so it is
asserted over generated cases rather than over a handful of examples.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from spine.splitting import (
    GroupTemporalSplit,
    RollingOriginSplit,
    temporal_train_test_split,
)

# ------------------------------------------------------- rolling origin, by example


def test_expanding_window_folds_are_laid_out_from_the_end():
    # n=10, 3 splits, horizon 2, step 2 -> test windows [4,5], [6,7], [8,9]
    splitter = RollingOriginSplit(n_splits=3, horizon=2)
    folds = list(splitter.split(np.arange(10)))
    assert [test.tolist() for _, test in folds] == [[4, 5], [6, 7], [8, 9]]
    # Expanding: every fold trains on everything before its test window.
    assert [train.tolist() for train, _ in folds] == [
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]


def test_sliding_window_keeps_the_training_length_fixed():
    splitter = RollingOriginSplit(n_splits=3, horizon=2, window="sliding", window_size=3)
    folds = list(splitter.split(np.arange(10)))
    assert [train.tolist() for train, _ in folds] == [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
    assert all(len(train) == 3 for train, _ in folds)


def test_gap_embargoes_observations_between_train_and_test():
    # With gap=2 the two observations immediately before each test window are
    # dropped from training, because a lag-2 feature would otherwise carry them in.
    splitter = RollingOriginSplit(n_splits=2, horizon=2, gap=2)
    folds = list(splitter.split(np.arange(10)))
    assert [test.tolist() for _, test in folds] == [[6, 7], [8, 9]]
    assert [train.tolist() for train, _ in folds] == [
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5],
    ]


def test_step_controls_overlap_between_test_windows():
    # The last fold always ends at the series end, so a step below the horizon
    # walks the windows backwards from there and they overlap.
    splitter = RollingOriginSplit(n_splits=3, horizon=3, step=1)
    folds = list(splitter.split(np.arange(10)))
    assert [test.tolist() for _, test in folds] == [[5, 6, 7], [6, 7, 8], [7, 8, 9]]


def test_get_n_splits_matches_the_number_of_folds_produced():
    splitter = RollingOriginSplit(n_splits=4, horizon=2)
    assert splitter.get_n_splits() == 4
    assert len(list(splitter.split(np.arange(50)))) == 4


def test_splitter_accepts_a_dataframe():
    frame = pd.DataFrame({"x": np.arange(20), "y": np.arange(20)})
    folds = list(RollingOriginSplit(n_splits=2, horizon=3).split(frame))
    assert len(folds) == 2
    assert folds[-1][1].tolist() == [17, 18, 19]


def test_splitter_raises_when_the_series_is_too_short():
    with pytest.raises(ValueError, match="needs at least"):
        list(RollingOriginSplit(n_splits=5, horizon=10).split(np.arange(20)))


def test_sliding_window_requires_a_window_size():
    with pytest.raises(ValueError, match="window_size"):
        RollingOriginSplit(n_splits=2, horizon=2, window="sliding")


def test_unknown_window_is_rejected():
    with pytest.raises(ValueError, match="window"):
        RollingOriginSplit(n_splits=2, horizon=2, window="rolling")


def test_negative_configuration_is_rejected():
    with pytest.raises(ValueError, match="horizon"):
        RollingOriginSplit(n_splits=2, horizon=0)
    with pytest.raises(ValueError, match="n_splits"):
        RollingOriginSplit(n_splits=0, horizon=2)
    with pytest.raises(ValueError, match="gap"):
        RollingOriginSplit(n_splits=2, horizon=2, gap=-1)


def test_splitter_is_usable_by_sklearn_cross_validate():
    """The splitter has to drop into sklearn unchanged, or nobody will use it."""
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import cross_val_score

    x = np.arange(60).reshape(-1, 1).astype(float)
    y = x.ravel() * 2.0
    scores = cross_val_score(DummyRegressor(), x, y, cv=RollingOriginSplit(n_splits=3, horizon=5))
    assert len(scores) == 3


# --------------------------------------------------------- the leak-safety property


@settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_samples=st.integers(min_value=10, max_value=400),
    n_splits=st.integers(min_value=1, max_value=6),
    horizon=st.integers(min_value=1, max_value=12),
    gap=st.integers(min_value=0, max_value=6),
    step=st.integers(min_value=1, max_value=6),
    window_size=st.integers(min_value=1, max_value=40),
    sliding=st.booleans(),
)
def test_no_fold_ever_trains_on_data_from_its_own_future(
    n_samples, n_splits, horizon, gap, step, window_size, sliding
):
    """For every generated configuration: max(train) + gap < min(test).

    This is the property that a random split violates and that inflates measured
    performance on temporal data. It must be impossible to configure around.
    """
    try:
        splitter = RollingOriginSplit(
            n_splits=n_splits,
            horizon=horizon,
            step=step,
            gap=gap,
            window="sliding" if sliding else "expanding",
            window_size=window_size if sliding else None,
        )
        folds = list(splitter.split(np.arange(n_samples)))
    except ValueError:
        # Not every generated combination fits in the generated series length.
        # Refusing loudly is correct behaviour, not a property violation.
        return

    assert len(folds) == n_splits
    seen_test_windows = []

    for train, test in folds:
        assert len(test) == horizon
        assert len(train) > 0

        # The guarantee.
        assert train.max() + gap < test.min()

        # No index is ever on both sides of the boundary.
        assert not set(train.tolist()) & set(test.tolist())

        # Both sides are contiguous, sorted, and inside the series.
        for side in (train, test):
            assert np.array_equal(side, np.sort(side))
            assert np.array_equal(np.diff(side), np.ones(len(side) - 1, dtype=int))
            assert side.min() >= 0
            assert side.max() < n_samples

        if sliding:
            assert len(train) <= window_size

        seen_test_windows.append(test)

    # The origin only ever moves forward.
    starts = [test.min() for test in seen_test_windows]
    assert starts == sorted(starts)


@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=20, max_value=200),
    n_splits=st.integers(min_value=2, max_value=5),
    horizon=st.integers(min_value=1, max_value=8),
)
def test_non_overlapping_test_windows_are_disjoint_by_default(n_samples, n_splits, horizon):
    """With the default step the test windows tile the end of the series."""
    try:
        folds = list(
            RollingOriginSplit(n_splits=n_splits, horizon=horizon).split(np.arange(n_samples))
        )
    except ValueError:
        return

    covered: set[int] = set()
    for _, test in folds:
        assert not covered & set(test.tolist())
        covered |= set(test.tolist())


# -------------------------------------------------------------- group-aware temporal


def test_group_temporal_split_never_cuts_a_timestamp_in_half():
    # Four rows per day for six days. A positional split would put part of a day
    # in train and the rest in test; this must not.
    days = np.repeat(pd.date_range("2026-01-01", periods=6, freq="D"), 4)
    folds = list(GroupTemporalSplit(n_splits=2, horizon=1).split(days))

    for train, test in folds:
        assert not set(days[train]) & set(days[test])
        assert len(test) == 4  # one whole day
    assert set(days[folds[-1][1]]) == {pd.Timestamp("2026-01-06")}


def test_group_temporal_split_respects_the_embargo_in_time_units():
    days = np.repeat(pd.date_range("2026-01-01", periods=8, freq="D"), 2)
    train, test = next(iter(GroupTemporalSplit(n_splits=1, horizon=1, gap=2).split(days)))
    # Test day is the 8th; gap of 2 days embargoes the 6th and 7th.
    assert set(days[test]) == {pd.Timestamp("2026-01-08")}
    assert max(days[train]) == pd.Timestamp("2026-01-05")


def test_group_temporal_split_drops_entities_that_straddle_the_boundary():
    # Customer "a" appears on both sides. Keeping them in test would let a model
    # score itself on someone it has already seen.
    days = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-04"])
    entities = np.array(["a", "b", "c", "a", "d"])
    train, test = next(
        iter(GroupTemporalSplit(n_splits=1, horizon=1).split(days, entities=entities))
    )
    assert set(entities[train]) == {"a", "b", "c"}
    assert set(entities[test]) == {"d"}  # "a" removed from test, not from train


def test_group_temporal_split_keeps_every_entity_when_none_straddle():
    days = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    entities = np.array(["a", "b", "c"])
    train, test = next(
        iter(GroupTemporalSplit(n_splits=1, horizon=1).split(days, entities=entities))
    )
    assert entities[test].tolist() == ["c"]


def test_group_temporal_split_rejects_a_length_mismatch():
    days = pd.to_datetime(["2026-01-01", "2026-01-02"])
    with pytest.raises(ValueError, match="same length"):
        list(GroupTemporalSplit(n_splits=1, horizon=1).split(days, entities=np.array(["a"])))


# ------------------------------------------------------------- single-cutoff split


def test_temporal_train_test_split_puts_the_cutoff_row_in_test():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "value": range(5),
        }
    )
    train, test = temporal_train_test_split(frame, "date", "2026-01-04")
    assert train["value"].tolist() == [0, 1, 2]
    assert test["value"].tolist() == [3, 4]


def test_temporal_train_test_split_sorts_unsorted_input():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-01", "2026-01-03"]),
            "value": [5, 1, 3],
        }
    )
    train, test = temporal_train_test_split(frame, "date", "2026-01-03")
    assert train["value"].tolist() == [1]
    assert test["value"].tolist() == [3, 5]


def test_temporal_train_test_split_refuses_an_empty_side():
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3), "value": range(3)})
    with pytest.raises(ValueError, match="empty"):
        temporal_train_test_split(frame, "date", "2027-01-01")


def test_temporal_train_test_split_rejects_a_missing_column():
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3)})
    with pytest.raises(KeyError, match="when"):
        temporal_train_test_split(frame, "when", "2026-01-02")
