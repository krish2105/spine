"""Dataset loading with schema validation.

Holds: resolution of the shared ``DATA_ROOT``, a table schema declaration, and a
validating reader that reports every violation at once rather than failing on the
first one.

Deliberately contains no per-dataset loaders. Eight datasets across four repos,
each needing different columns, is not a shared concern — it is four private
ones. What is shared is the contract: declare what a table must look like, and
find out immediately and completely when it does not.

Validation is intentionally shallow. It catches the failures that quietly corrupt
an analysis — a column that arrived as text, a key that is not unique, a date
column that is not in order — and it does not attempt to be a data quality
framework.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as pdtypes

__all__ = ["SchemaError", "TableSchema", "data_root", "read_table", "validate"]

_DTYPE_CHECKS = {
    "int": pdtypes.is_integer_dtype,
    "float": pdtypes.is_float_dtype,
    "str": pdtypes.is_string_dtype,
    "bool": pdtypes.is_bool_dtype,
    "datetime": pdtypes.is_datetime64_any_dtype,
    "category": isinstance,
}


class SchemaError(ValueError):
    """Raised when a table does not match its declared schema."""


@dataclass(frozen=True)
class TableSchema:
    """What a table must look like.

    Extra columns are allowed and unreported. A schema states what the analysis
    depends on; a dataset carrying additional columns is not a problem, whereas a
    dataset missing one is.

    Parameters
    ----------
    columns : mapping of str to str
        Column name to declared type, one of ``"int"``, ``"float"``, ``"str"``,
        ``"bool"``, ``"datetime"`` or ``"category"``.
    non_null : sequence of str, optional
        Columns that must contain no missing values.
    unique : sequence of str, optional
        Columns that together form a unique key.
    monotonic : str, optional
        Column that must be non-decreasing *across the whole table*. Declare the
        time column here: a table that arrived out of order will otherwise pass
        every other check and then silently break a temporal split. Note that a
        panel sorted by entity and then by date will fail this, correctly — sort
        a panel by time before declaring it.

    Raises
    ------
    ValueError
        If a declared type is not recognised.

    Examples
    --------
    >>> schema = TableSchema(
    ...     columns={"store": "int", "date": "datetime"},
    ...     unique=("store", "date"),
    ...     monotonic="date",
    ... )
    >>> schema.columns["store"]
    'int'
    """

    columns: Mapping[str, str]
    non_null: Sequence[str] = field(default_factory=tuple)
    unique: Sequence[str] = field(default_factory=tuple)
    monotonic: str | None = None

    def __post_init__(self) -> None:
        """Reject declared types this module cannot check.

        Raises
        ------
        ValueError
            If a declared type is not recognised.
        """
        unknown = {
            name: declared
            for name, declared in self.columns.items()
            if declared not in _DTYPE_CHECKS
        }
        if unknown:
            raise ValueError(
                f"unknown declared types {unknown}; supported types are {sorted(_DTYPE_CHECKS)}"
            )


def validate(frame: pd.DataFrame, schema: TableSchema) -> list[str]:
    """Report every way a frame fails its schema.

    Returns all violations rather than raising on the first, so a malformed
    extract is diagnosed in one read instead of one error per fix-and-rerun.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table to check.
    schema : TableSchema
        The declared schema.

    Returns
    -------
    list of str
        Human-readable violations. Empty when the frame conforms.

    Examples
    --------
    >>> import pandas as pd
    >>> schema = TableSchema(columns={"store": "int"}, non_null=("store",))
    >>> validate(pd.DataFrame({"store": ["a", "b"]}), schema)
    ["column 'store' is declared int but arrived as str"]
    """
    problems: list[str] = []

    for name, declared in schema.columns.items():
        if name not in frame.columns:
            problems.append(f"missing column {name!r} (declared {declared})")
            continue
        if declared == "category":
            if not isinstance(frame[name].dtype, pd.CategoricalDtype):
                problems.append(
                    f"column {name!r} is declared category but arrived as {frame[name].dtype}"
                )
        elif not _DTYPE_CHECKS[declared](frame[name]):
            problems.append(
                f"column {name!r} is declared {declared} but arrived as {frame[name].dtype}"
            )

    for name in schema.non_null:
        if name in frame.columns:
            missing = int(frame[name].isna().sum())
            if missing:
                problems.append(f"column {name!r} has {missing} missing values")

    key = [name for name in schema.unique if name in frame.columns]
    if key and len(key) == len(schema.unique):
        duplicated = int(frame.duplicated(subset=key).sum())
        if duplicated:
            problems.append(f"key {tuple(schema.unique)} has {duplicated} duplicate rows")

    if schema.monotonic and schema.monotonic in frame.columns:
        column = frame[schema.monotonic]
        if not column.is_monotonic_increasing:
            problems.append(
                f"column {schema.monotonic!r} is declared monotonic but is not sorted "
                f"ascending; a temporal split on this frame would be meaningless"
            )

    return problems


def read_table(
    path: str | Path,
    schema: TableSchema | None = None,
    **read_kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV or Parquet file and validate it against a schema.

    Parameters
    ----------
    path : str or pathlib.Path
        File to read. The extension selects the reader.
    schema : TableSchema, optional
        Schema to validate against. When omitted, the file is read unchecked.
    **read_kwargs
        Passed through to ``pandas.read_csv`` or ``pandas.read_parquet``.

    Returns
    -------
    pandas.DataFrame
        The validated table.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the extension is not ``.csv`` or ``.parquet``.
    SchemaError
        If the table violates the schema. The message lists every violation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, **read_kwargs)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path, **read_kwargs)
    else:
        raise ValueError(f"cannot read {suffix!r} files; read_table handles .csv and .parquet")

    if schema is not None:
        problems = validate(frame, schema)
        if problems:
            raise SchemaError(
                f"{path.name} does not match its schema:\n"
                + "\n".join(f"  - {problem}" for problem in problems)
            )
    return frame


def data_root(start: str | Path | None = None) -> Path:
    """Resolve the shared data directory.

    Looks first at the ``DATA_ROOT`` environment variable, then for a ``.env``
    file in ``start`` and its parents. Data lives outside the repositories and is
    never committed, so every repo needs the same answer to "where is it", and
    hard-coding that answer in five places guarantees it drifts.

    Parameters
    ----------
    start : str or pathlib.Path, optional
        Directory to begin the search from. Defaults to the working directory.

    Returns
    -------
    pathlib.Path
        The resolved data directory.

    Raises
    ------
    RuntimeError
        If ``DATA_ROOT`` is not configured anywhere, or points at a directory
        that does not exist.
    """
    configured = os.environ.get("DATA_ROOT")

    if configured is None:
        directory = Path(start) if start is not None else Path.cwd()
        for candidate in [directory, *directory.parents]:
            env_file = candidate / ".env"
            if not env_file.is_file():
                continue
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("DATA_ROOT="):
                    configured = line.split("=", 1)[1].strip().strip("'\"")
                    break
            if configured is not None:
                break

    if configured is None:
        raise RuntimeError(
            "DATA_ROOT is not configured; set it in the environment or in a .env "
            "file (see .env.example)"
        )

    resolved = Path(configured)
    if not resolved.is_dir():
        raise RuntimeError(f"DATA_ROOT points at {resolved}, which does not exist")
    return resolved
