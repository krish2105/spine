"""Shared input guards.

Private. Every public function in SPINE validates its inputs through here so that
a shape or NaN mistake surfaces as a named error at the call site rather than as a
plausible-looking number three steps later.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Coerce to a 1-D float array, rejecting NaN and infinity.

    Parameters
    ----------
    values : array-like
        The input to coerce.
    name : str
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        A 1-D array of dtype float64.

    Raises
    ------
    ValueError
        If the input is not 1-D, is empty, or contains NaN or infinity.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def same_length(*arrays: tuple[str, NDArray[np.float64]]) -> None:
    """Assert that every named array has the same length.

    Parameters
    ----------
    *arrays : tuple of (str, numpy.ndarray)
        Name and array pairs to compare.

    Raises
    ------
    ValueError
        If the lengths differ.
    """
    lengths = {name: len(array) for name, array in arrays}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{name}={n}" for name, n in lengths.items())
        raise ValueError(f"inputs must have the same length: {detail}")


def as_probabilities(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Coerce to a 1-D float array constrained to the closed unit interval.

    Parameters
    ----------
    values : array-like
        The input to coerce.
    name : str
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        A 1-D array of probabilities.

    Raises
    ------
    ValueError
        If any value falls outside [0, 1].
    """
    array = as_1d(values, name)
    if array.min() < 0.0 or array.max() > 1.0:
        raise ValueError(
            f"{name} must be probabilities in [0, 1], got range [{array.min()}, {array.max()}]"
        )
    return array


def as_binary(values: ArrayLike, name: str) -> NDArray[np.int_]:
    """Coerce to a 1-D integer array of zeros and ones.

    Parameters
    ----------
    values : array-like
        The input to coerce.
    name : str
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        A 1-D array containing only 0 and 1.

    Raises
    ------
    ValueError
        If any value is not 0 or 1.
    """
    array = as_1d(values, name)
    if not np.isin(array, (0.0, 1.0)).all():
        unique = np.unique(array)[:5]
        raise ValueError(f"{name} must be binary (0 or 1), found values {unique}")
    return array.astype(int)
