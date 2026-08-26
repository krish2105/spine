"""Tests for spine.cards.

The renderer is deliberately strict. Two of these tests exist because of academic
integrity rules rather than because of a software requirement: a metric with no
split attached is not interpretable, and a sensitive attribute that does not
declare whether it is a proxy invites the reader to assume it is not one.
"""

import textwrap

import pytest
import yaml

from spine.cards import CardValidationError, render_card, validate_card

VALID = {
    "model": {
        "name": "Credit challenger",
        "version": "0.3.1",
        "owner": "Krishna Mathur",
        "date": "2026-08-27",
    },
    "intended_use": "Rank consumer credit applications for manual review.",
    "data": {
        "source": "UCI German Credit",
        "provenance": "real",
        "rows": 1000,
    },
    "metrics": [
        {"name": "PR-AUC", "value": 0.412, "split": "holdout, applications after 2026-01-01"},
        {"name": "Brier score", "value": 0.081, "split": "holdout, applications after 2026-01-01"},
    ],
    "limitations": [
        "Public dataset; the distribution differs materially from UAE consumer data.",
    ],
}


def test_a_complete_card_validates_and_renders():
    assert validate_card(VALID) == []
    markdown = render_card(VALID)
    assert markdown.startswith("# Model card: Credit challenger")
    assert "0.3.1" in markdown
    assert "PR-AUC" in markdown
    assert "holdout, applications after 2026-01-01" in markdown


def test_render_reports_every_metric_next_to_its_split():
    markdown = render_card(VALID)
    metrics_table = markdown.split("## Metrics")[1]
    for line in ("PR-AUC", "Brier score"):
        row = next(r for r in metrics_table.splitlines() if line in r)
        assert "holdout, applications after 2026-01-01" in row


def test_a_metric_without_a_split_is_rejected():
    """A number with no split attached is not a result, it is a rumour."""
    spec = {**VALID, "metrics": [{"name": "PR-AUC", "value": 0.9}]}
    problems = validate_card(spec)
    assert any("split" in problem for problem in problems)
    with pytest.raises(CardValidationError, match="split"):
        render_card(spec)


def test_a_fairness_section_without_an_attribute_status_is_rejected():
    """A proxy attribute presented without that word invites a false reading."""
    spec = {**VALID, "fairness": {"attribute": "employment_type", "disparity_ratio": 0.78}}
    problems = validate_card(spec)
    assert any("attribute_status" in problem for problem in problems)
    with pytest.raises(CardValidationError, match="attribute_status"):
        render_card(spec)


def test_an_unrecognised_attribute_status_is_rejected():
    spec = {
        **VALID,
        "fairness": {
            "attribute": "employment_type",
            "attribute_status": "sort of protected",
            "disparity_ratio": 0.78,
        },
    }
    assert any("attribute_status" in problem for problem in validate_card(spec))


def test_a_proxy_attribute_is_labelled_in_the_rendered_card():
    spec = {
        **VALID,
        "fairness": {
            "attribute": "employment_type",
            "attribute_status": "proxy",
            "disparity_ratio": 0.78,
        },
    }
    markdown = render_card(spec)
    assert "proxy" in markdown.lower()
    assert "not a protected attribute" in markdown.lower()


def test_synthetic_data_is_caveated_at_the_top_of_the_card():
    spec = {**VALID, "data": {**VALID["data"], "provenance": "synthetic"}}
    markdown = render_card(spec)
    caveat_zone = markdown.split("## ")[0]
    assert "synthetic" in caveat_zone.lower()


def test_real_data_gets_no_caveat_banner():
    markdown = render_card(VALID)
    assert "synthetic" not in markdown.split("## ")[0].lower()


def test_an_unrecognised_provenance_is_rejected():
    spec = {**VALID, "data": {**VALID["data"], "provenance": "probably fine"}}
    assert any("provenance" in problem for problem in validate_card(spec))


def test_missing_required_sections_are_all_reported_at_once():
    problems = validate_card({"model": {"name": "x"}})
    joined = " ".join(problems)
    for expected in ("version", "owner", "intended_use", "data", "metrics", "limitations"):
        assert expected in joined


def test_a_card_can_be_loaded_from_yaml_text():
    text = textwrap.dedent(
        """
        model:
          name: Demand forecaster
          version: "1.0"
          owner: Krishna Mathur
          date: 2026-08-27
        intended_use: Weekly store-level demand forecasts feeding an ordering policy.
        data:
          source: Rossmann store sales
          provenance: real
        metrics:
          - name: MASE
            value: 0.83
            split: rolling origin, 3 folds, horizon 28
        limitations:
          - Single-period newsvendor; no lead time or lot sizing.
        """
    )
    markdown = render_card(yaml.safe_load(text))
    assert "Demand forecaster" in markdown
    assert "rolling origin, 3 folds, horizon 28" in markdown


def test_render_rejects_a_non_mapping():
    with pytest.raises(CardValidationError, match="mapping"):
        render_card(["not", "a", "card"])
