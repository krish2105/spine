"""Tests for spine.io."""

import pandas as pd
import pytest

from spine.io import SchemaError, TableSchema, data_root, read_table, validate

SCHEMA = TableSchema(
    columns={"store": "int", "date": "datetime", "sales": "float", "open": "bool"},
    non_null=("store", "date", "sales"),
    unique=("store", "date"),
    monotonic="date",
)


def _good_frame() -> pd.DataFrame:
    # Sorted by date, not by store: `monotonic` is a whole-column claim, so a
    # panel laid out entity-major would fail it, and should.
    return pd.DataFrame(
        {
            "store": [1, 2, 1],
            "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "sales": [100.0, 90.0, 120.0],
            "open": [True, False, True],
        }
    )


def test_a_conforming_frame_reports_no_violations():
    assert validate(_good_frame(), SCHEMA) == []


def test_every_violation_is_reported_at_once():
    """One error per read, not one error per fix-and-rerun cycle."""
    frame = _good_frame()
    frame = frame.drop(columns=["open"])  # missing column
    frame.loc[0, "sales"] = None  # null in a non-null column
    frame["store"] = frame["store"].astype(str)  # wrong dtype
    frame.loc[2, "date"] = pd.Timestamp("2020-01-01")  # breaks monotonicity

    problems = validate(frame, SCHEMA)
    joined = " ".join(problems)
    assert "open" in joined
    assert "sales" in joined
    assert "store" in joined
    assert "date" in joined
    assert len(problems) >= 4


def test_duplicate_keys_are_reported_with_a_count():
    frame = _good_frame()
    frame.loc[2, ["store", "date"]] = [2, pd.Timestamp("2026-01-01")]
    problems = validate(frame, SCHEMA)
    assert any("duplicate" in problem for problem in problems)


def test_extra_columns_are_allowed_and_not_reported():
    frame = _good_frame()
    frame["promo"] = 1
    assert validate(frame, SCHEMA) == []


def test_a_panel_sorted_by_entity_fails_a_monotonic_time_declaration():
    """Correctly. Entity-major order is the layout that breaks a temporal split."""
    frame = pd.DataFrame(
        {
            "store": [1, 1, 2],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01"]),
        }
    )
    schema = TableSchema(columns={"store": "int", "date": "datetime"}, monotonic="date")
    assert any("monotonic" in problem for problem in validate(frame, schema))


def test_monotonicity_is_only_checked_when_declared():
    schema = TableSchema(columns={"date": "datetime"})
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-02-01", "2026-01-01"])})
    assert validate(frame, schema) == []


def test_read_table_validates_a_csv(tmp_path):
    path = tmp_path / "sales.csv"
    _good_frame().to_csv(path, index=False)
    schema = TableSchema(columns={"store": "int", "sales": "float"}, non_null=("store", "sales"))
    frame = read_table(path, schema, parse_dates=["date"])
    assert len(frame) == 3


def test_read_table_raises_naming_every_violation(tmp_path):
    path = tmp_path / "broken.csv"
    pd.DataFrame({"store": ["a", "b"]}).to_csv(path, index=False)
    schema = TableSchema(columns={"store": "int", "sales": "float"})

    with pytest.raises(SchemaError) as caught:
        read_table(path, schema)
    message = str(caught.value)
    assert "sales" in message
    assert "store" in message
    assert "broken.csv" in message


def test_read_table_reads_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "sales.parquet"
    _good_frame().to_parquet(path)
    assert len(read_table(path, SCHEMA)) == 3


def test_read_table_rejects_an_unknown_extension(tmp_path):
    path = tmp_path / "sales.xlsx"
    path.write_text("not really a spreadsheet")
    with pytest.raises(ValueError, match="\\.xlsx"):
        read_table(path, None)


def test_read_table_reports_a_missing_file_with_its_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere.csv"):
        read_table(tmp_path / "nowhere.csv", None)


def test_read_table_without_a_schema_skips_validation(tmp_path):
    path = tmp_path / "anything.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(path, index=False)
    assert len(read_table(path, None)) == 2


def test_unknown_declared_dtype_is_rejected():
    with pytest.raises(ValueError, match="timestamp"):
        TableSchema(columns={"date": "timestamp"})


# ---------------------------------------------------------------------- data root


def test_data_root_prefers_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert data_root() == tmp_path


def test_data_root_falls_back_to_a_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    target = tmp_path / "shared-data"
    target.mkdir()
    (tmp_path / ".env").write_text(f"# a comment\nDATA_ROOT={target}\n")
    assert data_root(start=tmp_path) == target


def test_data_root_searches_parent_directories_for_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    target = tmp_path / "shared-data"
    target.mkdir()
    (tmp_path / ".env").write_text(f"DATA_ROOT={target}\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert data_root(start=nested) == target


def test_data_root_raises_when_it_is_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="DATA_ROOT"):
        data_root(start=tmp_path)


def test_data_root_raises_when_it_points_somewhere_that_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="does not exist"):
        data_root()
