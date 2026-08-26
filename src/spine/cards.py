"""Model card generator: YAML in, markdown out.

Deliberately regulation-blind. It validates structure and renders markdown; it
knows nothing about CBUAE, DIFC Regulation 10 or the EU AI Act. Clause-level
mapping belongs in one place, and that place is MIZAN.

Two structural rules it enforces by refusing to render, both of them integrity
rules rather than software requirements:

Every metric carries the split it was computed on.
    A number without its split is not a result. "AUC 0.84" is unfalsifiable;
    "AUC 0.84 on applications after 2026-01-01, held out" can be checked.

Every sensitive attribute declares whether it is a proxy.
    Where the real attribute is unavailable and something correlated stands in,
    the analysis is a methodological demonstration. Silence on that point reads
    as a claim that it is not, so silence is rejected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["CardValidationError", "render_card", "validate_card"]

_PROVENANCE = ("real", "synthetic", "simulated")
_ATTRIBUTE_STATUS = ("proxy", "protected")
_REQUIRED_MODEL_KEYS = ("name", "version", "owner")
_REQUIRED_TOP_LEVEL = ("model", "intended_use", "data", "metrics", "limitations")


class CardValidationError(ValueError):
    """Raised when a card specification is not renderable."""


def validate_card(spec: Any) -> list[str]:
    """Report everything wrong with a card specification.

    Returns every problem at once rather than the first one, so a card is fixed
    in one pass instead of one error per rerun.

    Parameters
    ----------
    spec : mapping
        The card specification, typically ``yaml.safe_load`` of a card file.

    Returns
    -------
    list of str
        Human-readable problems. Empty when the card is renderable.

    Examples
    --------
    Every problem comes back together, not one per rerun:

    >>> problems = validate_card({"model": {"name": "m"}})
    >>> len(problems)
    6
    >>> [p for p in problems if "version" in p]
    ["model is missing 'version'"]
    """
    if not isinstance(spec, Mapping):
        return [f"a card must be a mapping, got {type(spec).__name__}"]

    problems = [
        f"missing top-level section '{key}'" for key in _REQUIRED_TOP_LEVEL if key not in spec
    ]

    model = spec.get("model")
    if isinstance(model, Mapping):
        problems += [
            f"model is missing {key!r}" for key in _REQUIRED_MODEL_KEYS if not model.get(key)
        ]
    elif "model" in spec:
        problems.append("model must be a mapping")

    data = spec.get("data")
    if isinstance(data, Mapping):
        if not data.get("source"):
            problems.append("data is missing 'source'")
        provenance = data.get("provenance")
        if provenance not in _PROVENANCE:
            problems.append(f"data.provenance must be one of {_PROVENANCE}, got {provenance!r}")
    elif "data" in spec:
        problems.append("data must be a mapping")

    metrics = spec.get("metrics")
    if isinstance(metrics, Sequence) and not isinstance(metrics, str):
        for position, metric in enumerate(metrics, start=1):
            if not isinstance(metric, Mapping):
                problems.append(f"metric {position} must be a mapping")
                continue
            for key in ("name", "value", "split"):
                if metric.get(key) in (None, ""):
                    problems.append(
                        f"metric {position} ({metric.get('name', 'unnamed')}) is missing {key!r}"
                    )
    elif "metrics" in spec:
        problems.append("metrics must be a list")

    fairness = spec.get("fairness")
    if isinstance(fairness, Mapping):
        if not fairness.get("attribute"):
            problems.append("fairness is missing 'attribute'")
        if fairness.get("attribute_status") not in _ATTRIBUTE_STATUS:
            problems.append(
                f"fairness.attribute_status must be one of {_ATTRIBUTE_STATUS}, got "
                f"{fairness.get('attribute_status')!r}; a proxy is not a protected "
                f"attribute and the card must say which this is"
            )
    elif "fairness" in spec:
        problems.append("fairness must be a mapping")

    return problems


def _section(title: str, body: str) -> str:
    """Format one markdown section.

    Parameters
    ----------
    title : str
        Section heading.
    body : str
        Section content.

    Returns
    -------
    str
        The rendered section, with trailing blank line.
    """
    return f"## {title}\n\n{body}\n"


def _bullets(items: Any) -> str:
    """Render a value as a markdown bullet list.

    Parameters
    ----------
    items : object
        A sequence of items, or a single scalar.

    Returns
    -------
    str
        Markdown bullets, one per item.
    """
    if isinstance(items, str) or not isinstance(items, Sequence):
        items = [items]
    return "\n".join(f"- {item}" for item in items)


def render_card(spec: Any) -> str:
    """Render a validated card specification as markdown.

    Parameters
    ----------
    spec : mapping
        The card specification.

    Returns
    -------
    str
        The model card as markdown.

    Raises
    ------
    CardValidationError
        If the specification has any problem reported by :func:`validate_card`.
        The message lists all of them.

    Examples
    --------
    >>> card = {
    ...     "model": {"name": "Demo", "version": "1.0", "owner": "K. Mathur"},
    ...     "intended_use": "Illustration.",
    ...     "data": {"source": "fixture", "provenance": "real"},
    ...     "metrics": [{"name": "MASE", "value": 0.83, "split": "rolling origin"}],
    ...     "limitations": ["Not a real model."],
    ... }
    >>> render_card(card).splitlines()[0]
    '# Model card: Demo'
    """
    problems = validate_card(spec)
    if problems:
        raise CardValidationError(
            "this card cannot be rendered:\n" + "\n".join(f"  - {p}" for p in problems)
        )

    model, data = spec["model"], spec["data"]
    parts = [f"# Model card: {model['name']}\n"]

    if data["provenance"] != "real":
        parts.append(
            f"> **The data behind every number in this card is "
            f"{data['provenance']}.** Results demonstrate the method. They are not "
            f"findings about the real population.\n"
        )

    parts.append(
        _section(
            "Model",
            "\n".join(
                f"- **{key.replace('_', ' ').capitalize()}:** {value}"
                for key, value in model.items()
            ),
        )
    )
    parts.append(_section("Intended use", str(spec["intended_use"])))
    parts.append(
        _section(
            "Data",
            "\n".join(
                f"- **{key.replace('_', ' ').capitalize()}:** {value}"
                for key, value in data.items()
            ),
        )
    )

    rows = ["| Metric | Value | Split |", "| --- | --- | --- |"]
    rows += [f"| {m['name']} | {m['value']} | {m['split']} |" for m in spec["metrics"]]
    parts.append(_section("Metrics", "\n".join(rows)))

    fairness = spec.get("fairness")
    if fairness:
        lines = [
            f"- **Attribute:** {fairness['attribute']}",
            f"- **Attribute status:** {fairness['attribute_status']}",
        ]
        lines += [
            f"- **{key.replace('_', ' ').capitalize()}:** {value}"
            for key, value in fairness.items()
            if key not in ("attribute", "attribute_status")
        ]
        if fairness["attribute_status"] == "proxy":
            lines.append(
                f"\n`{fairness['attribute']}` is a **proxy**, not a protected "
                f"attribute. Every disparity below measures the proxy. Treating it "
                f"as evidence about the protected group it stands for is a "
                f"methodological demonstration, not a fairness finding."
            )
        parts.append(_section("Fairness", "\n".join(lines)))

    parts.append(_section("Limitations", _bullets(spec["limitations"])))
    return "\n".join(parts)
