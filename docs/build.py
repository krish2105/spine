"""Render docs/index.html from reports/proof.json and the library's own docstrings.

Nothing on the page is typed by a human. Numbers come from the proof run;
signatures, summaries and worked examples come from ``inspect`` over the
installed package. A claim on the page and the code it describes cannot drift
apart, because one is generated from the other.

Run: uv run python docs/build.py
"""

from __future__ import annotations

import html
import importlib
import inspect
import json
import math
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "reports" / "proof.json"
OUTPUT = ROOT / "docs" / "index.html"

MODULES = [
    ("spine.splitting", "Temporal and group-aware splitters. Nothing here shuffles."),
    ("spine.metrics", "Forecasting, classification and uplift metrics."),
    ("spine.decisions", "The bridge from a forecast or a probability to an action."),
    ("spine.fairness", "Group rates, disparity ratios, calibration by group."),
    ("spine.cards", "Model card generator. YAML in, markdown out."),
    ("spine.io", "Shared data root, schema declaration, validating reader."),
]


# --------------------------------------------------------------------- formatting


def esc(value: Any) -> str:
    """Escape a value for HTML text content.

    Parameters
    ----------
    value : object
        Anything renderable as a string.

    Returns
    -------
    str
        HTML-escaped text.
    """
    return html.escape(str(value), quote=True)


def decimal_parts(value: float, places: int = 6) -> tuple[str, str]:
    """Split a number into the parts either side of its decimal point.

    The page aligns every number in a column on the decimal point, which needs
    the two halves rendered separately.

    Parameters
    ----------
    value : float
        The number to split.
    places : int, default 6
        Digits after the point.

    Returns
    -------
    whole : str
        Digits before the point, sign included.
    fraction : str
        Digits after the point, without the point itself.
    """
    text = f"{value:.{places}f}"
    whole, _, fraction = text.partition(".")
    return whole, fraction


def format_delta(delta: float) -> str:
    """Format an absolute difference compactly.

    Parameters
    ----------
    delta : float
        The difference.

    Returns
    -------
    str
        ``"0"`` for an exact match, otherwise two significant figures in
        scientific notation.
    """
    return "0" if delta == 0 else f"{delta:.2e}"


def number_cell(value: float, places: int = 6) -> str:
    """Render a decimal-aligned number cell.

    Parameters
    ----------
    value : float
        The number.
    places : int, default 6
        Digits after the point.

    Returns
    -------
    str
        HTML for one aligned number.
    """
    whole, fraction = decimal_parts(value, places)
    return (
        f'<span class="num"><span class="num-w">{esc(whole)}</span>'
        f'<span class="num-p">.</span>'
        f'<span class="num-f">{esc(fraction)}</span></span>'
    )


# ------------------------------------------------------------------- svg plotting


def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    """Map a value from one range onto another.

    Parameters
    ----------
    value : float
        Input value.
    lo, hi : float
        Input range.
    out_lo, out_hi : float
        Output range.

    Returns
    -------
    float
        The mapped value.
    """
    if hi == lo:
        return out_lo
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def nice_ticks(top: float, target: int = 4) -> list[float]:
    """Choose round tick values between 0 and ``top``.

    Axis labels like 239, 478, 717 are arithmetically correct and useless. This
    picks a step from the 1-2-5 sequence so the labels read as round numbers.

    Parameters
    ----------
    top : float
        Upper bound of the axis.
    target : int, default 4
        Roughly how many intervals to aim for.

    Returns
    -------
    list of float
        Tick positions from 0 upwards, none exceeding ``top``.
    """
    if top <= 0:
        return [0.0]
    raw = top / target
    magnitude = 10 ** math.floor(math.log10(raw))
    # Pick the 1-2-5 step that lands closest to the requested number of
    # intervals, rather than the first one at or above the raw spacing: that
    # rounds up hard enough to leave an axis with a single labelled tick.
    candidates = [m * magnitude for m in (0.5, 1, 2, 2.5, 5, 10)]
    step = min(candidates, key=lambda c: abs(top / c - target))
    ticks, value = [], 0.0
    while value <= top + 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


def polyline(xs: list[float], ys: list[float]) -> str:
    """Join coordinates into an SVG points attribute.

    Parameters
    ----------
    xs, ys : list of float
        Coordinates.

    Returns
    -------
    str
        Space-separated ``x,y`` pairs.
    """
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys, strict=True))


def svg_rolling_origin(data: dict) -> str:
    """Draw the rolling-origin fold layout.

    This is the page's thesis in one picture: training in graphite, the embargo
    hatched, the test window in ink, and the origin moving forward one fold at a
    time without ever reaching back across itself.

    Parameters
    ----------
    data : dict
        The ``rolling_origin`` figure payload.

    Returns
    -------
    str
        Inline SVG.
    """
    n = data["n_samples"]
    folds = data["folds"]
    left, right = 44.0, 596.0
    top, row_height = 30.0, 30.0
    bar = 15.0
    height = top + len(folds) * row_height + 34

    def x_of(index: float) -> float:
        return scale(index, 0, n, left, right)

    parts = [
        f'<svg viewBox="0 0 640 {height:.0f}" role="img" '
        f'aria-label="Five rolling-origin folds over {n} observations. Each fold '
        f"trains on everything before an embargo gap and tests on the six "
        f"observations after it; no training window ever reaches past its own "
        f'test window." class="fig fig-hero">'
    ]

    for i, fold in enumerate(folds):
        y = top + i * row_height
        delay = f"{i * 0.04:.2f}s"
        parts.append(
            f'<text x="{left - 10:.1f}" y="{y + bar - 3:.1f}" class="svg-label-r">{i + 1}</text>'
        )
        parts.append(
            f'<g class="fold" style="--d:{delay}">'
            f'<rect x="{x_of(fold["train_start"]):.1f}" y="{y:.1f}" '
            f'width="{x_of(fold["train_end"]) - x_of(fold["train_start"]):.1f}" '
            f'height="{bar}" class="bar-train"/>'
            f'<rect x="{x_of(fold["train_end"]):.1f}" y="{y:.1f}" '
            f'width="{x_of(fold["test_start"]) - x_of(fold["train_end"]):.1f}" '
            f'height="{bar}" class="bar-gap"/>'
            f'<rect x="{x_of(fold["test_start"]):.1f}" y="{y:.1f}" '
            f'width="{x_of(fold["test_end"]) - x_of(fold["test_start"]):.1f}" '
            f'height="{bar}" class="bar-test"/>'
            f'<line x1="{x_of(fold["test_start"]):.1f}" y1="{y - 4:.1f}" '
            f'x2="{x_of(fold["test_start"]):.1f}" y2="{y + bar + 4:.1f}" '
            f'class="origin"/>'
            f"</g>"
        )

    axis_y = top + len(folds) * row_height + 6
    parts.append(
        f'<line x1="{left:.1f}" y1="{axis_y:.1f}" x2="{right:.1f}" y2="{axis_y:.1f}" class="axis"/>'
    )
    for tick in range(0, n + 1, 10):
        tx = x_of(tick)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{axis_y:.1f}" x2="{tx:.1f}" y2="{axis_y + 4:.1f}" '
            f'class="axis"/>'
            f'<text x="{tx:.1f}" y="{axis_y + 16:.1f}" class="svg-tick">{tick}</text>'
        )
    parts.append(
        f'<text x="{right:.1f}" y="{axis_y + 28:.1f}" class="svg-tick svg-tick-r">'
        f"observation index &#8594;</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _frame(width: float, height: float, label: str, css_class: str) -> tuple[list[str], dict]:
    """Start an SVG with a plot area and return the parts list and geometry.

    Parameters
    ----------
    width, height : float
        Canvas size in the viewBox.
    label : str
        Accessible description.
    css_class : str
        Class applied to the svg element.

    Returns
    -------
    parts : list of str
        SVG fragments so far.
    box : dict
        Plot-area bounds as ``left``, ``right``, ``top``, ``bottom``.
    """
    box = {"left": 52.0, "right": width - 14, "top": 14.0, "bottom": height - 40}
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{esc(label)}" class="fig {css_class}">'
    ]
    return parts, box


def _axes(parts: list[str], box: dict, x_ticks: list, y_ticks: list, x_title: str) -> None:
    """Draw axis lines, ticks and labels into an SVG under construction.

    Parameters
    ----------
    parts : list of str
        SVG fragments, appended in place.
    box : dict
        Plot-area bounds.
    x_ticks, y_ticks : list of tuple
        ``(position, label)`` pairs in canvas coordinates.
    x_title : str
        Caption for the x axis.
    """
    parts.append(
        f'<line x1="{box["left"]:.1f}" y1="{box["bottom"]:.1f}" '
        f'x2="{box["right"]:.1f}" y2="{box["bottom"]:.1f}" class="axis"/>'
    )
    for x, label in x_ticks:
        parts.append(
            f'<line x1="{x:.1f}" y1="{box["bottom"]:.1f}" x2="{x:.1f}" '
            f'y2="{box["bottom"] + 4:.1f}" class="axis"/>'
            f'<text x="{x:.1f}" y="{box["bottom"] + 16:.1f}" class="svg-tick">'
            f"{esc(label)}</text>"
        )
    for y, label in y_ticks:
        parts.append(
            f'<line x1="{box["left"]:.1f}" y1="{y:.1f}" x2="{box["right"]:.1f}" '
            f'y2="{y:.1f}" class="grid"/>'
            f'<text x="{box["left"] - 8:.1f}" y="{y + 4:.1f}" class="svg-label-r">'
            f"{esc(label)}</text>"
        )
    parts.append(
        f'<text x="{box["right"]:.1f}" y="{box["bottom"] + 32:.1f}" '
        f'class="svg-tick svg-tick-r">{esc(x_title)}</text>'
    )


def svg_calibration(data: dict) -> str:
    """Draw a reliability diagram from computed calibration bins.

    Parameters
    ----------
    data : dict
        The ``calibration`` figure payload.

    Returns
    -------
    str
        Inline SVG.
    """
    parts, box = _frame(
        420,
        300,
        "Reliability diagram. Ten bins of an over-confident fixture model; points "
        "sit below the diagonal at high predicted probability, meaning the model "
        "predicts more risk than occurs.",
        "fig-square",
    )

    def px(v: float) -> float:
        return scale(v, 0, 1, box["left"], box["right"])

    def py(v: float) -> float:
        return scale(v, 0, 1, box["bottom"], box["top"])

    _axes(
        parts,
        box,
        [(px(v), f"{v:g}") for v in (0, 0.25, 0.5, 0.75, 1)],
        [(py(v), f"{v:g}") for v in (0, 0.25, 0.5, 0.75, 1)],
        "predicted probability →",
    )
    parts.append(
        f'<line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(1):.1f}" y2="{py(1):.1f}" '
        f'class="reference"/>'
    )
    xs = [px(v) for v in data["mean_predicted"]]
    ys = [py(v) for v in data["observed_rate"]]
    parts.append(f'<polyline points="{polyline(xs, ys)}" class="series"/>')
    for x, y in zip(xs, ys, strict=True):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" class="point"/>')
    parts.append(
        f'<text x="{px(0.99):.1f}" y="{py(1.0) - 6:.1f}" '
        f'class="svg-note svg-note-end">perfect calibration</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def svg_qini(data: dict) -> str:
    """Draw the Qini curve against the random-targeting line.

    Parameters
    ----------
    data : dict
        The ``qini`` figure payload.

    Returns
    -------
    str
        Inline SVG.
    """
    parts, box = _frame(
        420,
        300,
        "Qini curve. Incremental responders rise steeply while the ranking is "
        "still finding persuadable people, then flatten onto the random line.",
        "fig-square",
    )
    top_gain = max(data["gain"])

    def px(v: float) -> float:
        return scale(v, 0, 1, box["left"], box["right"])

    def py(v: float) -> float:
        return scale(v, 0, top_gain, box["bottom"], box["top"])

    _axes(
        parts,
        box,
        [(px(v), f"{v:g}") for v in (0, 0.25, 0.5, 0.75, 1)],
        [(py(v), f"{v:,.0f}") for v in nice_ticks(top_gain)],
        "fraction targeted →",
    )
    parts.append(
        f'<line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(1):.1f}" '
        f'y2="{py(data["total_incremental"]):.1f}" class="reference"/>'
    )
    xs = [px(v) for v in data["fraction"]]
    ys = [py(v) for v in data["gain"]]
    parts.append(f'<polyline points="{polyline(xs, ys)}" class="series"/>')
    parts.append(
        f'<text x="{px(0.42):.1f}" y="{py(top_gain * 0.42):.1f}" class="svg-note">'
        f"random targeting</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def svg_pinball(data: dict) -> str:
    """Draw mean pinball loss per quantile for two fixture forecasters.

    Parameters
    ----------
    data : dict
        The ``pinball`` figure payload.

    Returns
    -------
    str
        Inline SVG.
    """
    parts, box = _frame(
        420,
        300,
        "Mean pinball loss at each quantile for two forecasters. They tie at the "
        "median, where both predict the same value, and the forecaster with the "
        "correct spread wins by a widening margin towards both tails.",
        "fig-square",
    )
    top = max(max(data["sharp"]), max(data["too_narrow"])) * 1.1

    def px(v: float) -> float:
        return scale(v, 0.05, 0.95, box["left"], box["right"])

    def py(v: float) -> float:
        return scale(v, 0, top, box["bottom"], box["top"])

    _axes(
        parts,
        box,
        [(px(v), f"{v:g}") for v in (0.1, 0.3, 0.5, 0.7, 0.9)],
        [(py(v), f"{v:g}") for v in nice_ticks(top, 3)],
        "quantile level →",
    )
    for key, css in (("too_narrow", "series-muted"), ("sharp", "series")):
        xs = [px(v) for v in data["levels"]]
        ys = [py(v) for v in data[key]]
        parts.append(f'<polyline points="{polyline(xs, ys)}" class="{css}"/>')
        for x, y in zip(xs, ys, strict=True):
            marker = "point" if key == "sharp" else "point-muted"
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" class="{marker}"/>')
    # A legend in the empty band under the curves, rather than labels sitting on
    # top of the lines they name.
    for offset, (css, note_css, label) in enumerate(
        [
            ("series-muted", "svg-note", "over-confident (sd 8)"),
            ("series", "svg-note svg-note-accent", "correct spread (sd 20)"),
        ]
    ):
        y = py(top * (0.20 - 0.09 * offset))
        parts.append(
            f'<line x1="{px(0.13):.1f}" y1="{y - 3:.1f}" x2="{px(0.20):.1f}" '
            f'y2="{y - 3:.1f}" class="{css}"/>'
            f'<text x="{px(0.22):.1f}" y="{y:.1f}" class="{note_css}">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_cost_curve(data: dict) -> str:
    """Draw expected cost against classification threshold.

    Parameters
    ----------
    data : dict
        The ``cost_curve`` figure payload.

    Returns
    -------
    str
        Inline SVG.
    """
    parts, box = _frame(
        420,
        300,
        "Expected cost per case against the decision threshold. The basin around "
        "the minimum is flat, so the exact cutoff matters less than the cost "
        "estimate that produced it.",
        "fig-square",
    )
    top = max(data["expected_cost"]) * 1.05

    def px(v: float) -> float:
        return scale(v, 0, 1, box["left"], box["right"])

    def py(v: float) -> float:
        return scale(v, 0, top, box["bottom"], box["top"])

    _axes(
        parts,
        box,
        [(px(v), f"{v:g}") for v in (0, 0.25, 0.5, 0.75, 1)],
        [(py(v), f"{v:g}") for v in nice_ticks(top, 3)],
        "threshold →",
    )
    xs = [px(v) for v in data["threshold"]]
    ys = [py(v) for v in data["expected_cost"]]
    parts.append(f'<polyline points="{polyline(xs, ys)}" class="series"/>')
    marker_x = px(data["optimal_threshold"])
    parts.append(
        f'<line x1="{marker_x:.1f}" y1="{box["top"]:.1f}" x2="{marker_x:.1f}" '
        f'y2="{box["bottom"]:.1f}" class="origin"/>'
        f'<text x="{marker_x + 6:.1f}" y="{box["top"] + 12:.1f}" '
        f'class="svg-note svg-note-accent">optimum '
        f"{data['optimal_threshold']:.3f}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------- api reference


def examples_block(docstring: str) -> str:
    """Extract the Examples section of a numpy-style docstring.

    Every example on this page is executed as a doctest by the test suite, so
    what the page shows is what the code actually does.

    Parameters
    ----------
    docstring : str
        The full docstring.

    Returns
    -------
    str
        The example lines, or an empty string when there is no Examples section.
    """
    lines = docstring.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Examples" and i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            body = textwrap.dedent("\n".join(lines[i + 2 :])).strip("\n")
            return body.rstrip()
    return ""


def summary_of(docstring: str) -> str:
    """Return the first line of a docstring.

    Parameters
    ----------
    docstring : str
        The full docstring.

    Returns
    -------
    str
        The summary line.
    """
    return docstring.strip().splitlines()[0] if docstring.strip() else ""


def api_entries() -> list[dict]:
    """Collect the public API by inspecting the installed package.

    Returns
    -------
    list of dict
        One entry per module, each holding its members.
    """
    entries = []
    for module_name, blurb in MODULES:
        module = importlib.import_module(module_name)
        members = []
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name)
            try:
                signature = str(inspect.signature(obj))
            except (TypeError, ValueError):
                signature = ""
            doc = inspect.getdoc(obj) or ""
            members.append(
                {
                    "name": name,
                    "signature": signature,
                    "summary": summary_of(doc),
                    "example": examples_block(doc),
                    "kind": "class" if inspect.isclass(obj) else "function",
                }
            )
        entries.append({"module": module_name, "blurb": blurb, "members": members})
    return entries


# ------------------------------------------------------------------- page render


def render(proof: dict) -> str:
    """Assemble the complete HTML page.

    Parameters
    ----------
    proof : dict
        The parsed ``reports/proof.json``.

    Returns
    -------
    str
        A standalone HTML document.
    """
    figures = proof["figures"]
    leak = proof["leak_safety"]
    verifications = proof["verifications"]
    passed = sum(1 for row in verifications if row["passed"])

    ledger = []
    for row in verifications:
        mark = "verified" if row["passed"] else "FAILED"
        tolerance = "any" if row["tolerance"] == float("inf") else format_delta(row["tolerance"])
        ledger.append(
            f'<div class="ledger-row{"" if row["passed"] else " failed"}">'
            f'<div class="ledger-claim">'
            f"<p>{esc(row['claim'])}</p>"
            f'<p class="ledger-source">{esc(row["source"])} '
            f'<span class="sep">against</span> {esc(row["expected_from"])}</p>'
            f"</div>"
            f'<div class="ledger-figs">'
            f'<div class="cell"><span class="cell-key">computed</span>'
            f"{number_cell(row['computed'])}</div>"
            f'<div class="cell"><span class="cell-key">expected</span>'
            f"{number_cell(row['expected'])}</div>"
            f'<div class="cell cell-delta"><span class="cell-key">&#916;</span>'
            f'<span class="num num-sci">{esc(format_delta(row["delta"]))}</span></div>'
            f'<div class="cell cell-tol"><span class="cell-key">tolerance</span>'
            f'<span class="num num-sci">{esc(tolerance)}</span></div>'
            f'<div class="cell cell-mark"><span class="mark">{esc(mark)}</span></div>'
            f"</div></div>"
        )

    api_html = []
    for entry in api_entries():
        members = "".join(
            f'<article class="member" id="{esc(entry["module"])}.{esc(m["name"])}">'
            f'<h3><span class="member-kind">{esc(m["kind"])}</span>'
            f"<code>{esc(m['name'])}</code>"
            f'<span class="member-sig">{esc(m["signature"])}</span></h3>'
            f"<p>{esc(m['summary'])}</p>"
            + (
                f'<pre class="example"><code>{esc(m["example"])}</code></pre>'
                if m["example"]
                else ""
            )
            + "</article>"
            for m in entry["members"]
        )
        api_html.append(
            f'<section class="api-module">'
            f"<h2><code>{esc(entry['module'])}</code></h2>"
            f'<p class="module-blurb">{esc(entry["blurb"])}</p>'
            f"{members}</section>"
        )

    packages = "".join(
        f"<li><span>{esc(name)}</span><span>{esc(ver)}</span></li>"
        for name, ver in proof["packages"].items()
    )

    calib, qini, pinball, cost = (
        figures["calibration"],
        figures["qini"],
        figures["pinball"],
        figures["cost_curve"],
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPINE</title>
<meta name="description" content="Evaluation, temporal splitting and decision logic
 shared by four MAIB Term 4 projects. Every number on this page is regenerated from a run.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Spectral:ital,wght@0,300;0,400;0,600;1,400&display=swap">
<style>{STYLE}</style>
</head>
<body>
<a class="skip" href="#verifications">Skip to the verifications</a>

<header class="masthead">
  <div class="wrap masthead-inner">
    <div class="wordmark">
      <span class="wordmark-name">SPINE</span>
      <span class="wordmark-meta">01 &middot; shared library</span>
    </div>
    <button id="theme" type="button" class="theme" aria-live="polite">
      <span class="theme-dot" aria-hidden="true"></span><span id="theme-label">Dark</span>
    </button>
  </div>
</header>

<main>
<section class="hero wrap">
  <p class="eyebrow">MAIB Term 4 &middot; SP Jain School of Global Management, Dubai</p>
  <h1>The split boundary, the MASE denominator,<br>and the critical fractile.</h1>
  <p class="lede">
    Four projects need the same three things: honest evaluation, temporal splitting,
    and decisions under asymmetric cost. SPINE holds those and nothing else &mdash;
    no models, no estimators, no framework. These are the parts that are easy to get
    subtly wrong and expensive to get wrong three times.
  </p>

  <figure class="hero-figure">
    {svg_rolling_origin(figures["rolling_origin"])}
    <figcaption>
      <div class="key">
        <span><i class="swatch swatch-train"></i>train</span>
        <span><i class="swatch swatch-gap"></i>embargo</span>
        <span><i class="swatch swatch-test"></i>test</span>
      </div>
      <p>
        <code>{esc(figures["rolling_origin"]["config"])}</code> over
        {figures["rolling_origin"]["n_samples"]} observations, drawn from the folds the
        splitter actually produced. The origin moves forward; no training window ever
        reaches past the embargo into its own test window.
      </p>
    </figcaption>
  </figure>

  <p class="assertion">
    <strong>{leak["cases"]:,}</strong> folds checked across every configuration in a
    deterministic grid, plus <strong>{leak["hypothesis_cases"]}</strong> randomised
    cases generated by hypothesis in the test suite.
    <strong>{leak["violations"]}</strong> boundary violations.
    <span class="assertion-rule"><code>{esc(leak["assertion"])}</code></span>
  </p>
</section>

<section class="wrap scope">
  <div class="scope-col">
    <h2>What is here</h2>
    <ul>
      <li>Rolling-origin splits, expanding and sliding, with an embargo for lagged features</li>
      <li>MASE scaled on the training window, pinball loss, PR-AUC, Brier,
          calibration error, Qini</li>
      <li>Newsvendor order quantities, safety stock, cost-sensitive thresholds</li>
      <li>Group rates, disparity ratios, calibration by group</li>
      <li>A model card renderer that refuses to omit a split or a proxy label</li>
    </ul>
  </div>
  <div class="scope-col scope-col-negative">
    <h2>What is not</h2>
    <ul>
      <li>Models. SPINE holds evaluation, not estimators.</li>
      <li>Random splits. Every split here is temporal or group-aware.</li>
      <li>MAPE. Undefined at zero, asymmetric, not comparable across scales.</li>
      <li>Regulatory logic. Clause mapping lives in MIZAN, in one place.</li>
      <li>Plotting. Functions return arrays; presentation is the caller's business.</li>
    </ul>
  </div>
</section>

<section class="wrap" id="verifications">
  <div class="section-head">
    <h2>Verifications</h2>
    <p class="section-note">
      {passed} of {len(verifications)} pass. Each row was computed by SPINE and checked
      against something that shares no code with it &mdash; the Python standard library,
      scikit-learn, Nixtla's utilsforecast, or arithmetic done by hand.
    </p>
  </div>
  <div class="ledger">{"".join(ledger)}</div>
</section>

<section class="wrap">
  <div class="section-head">
    <h2>Behaviour on fixtures</h2>
    <p class="section-note">
      Every figure below is drawn from seeded synthetic data, not from a dataset. They
      show that the functions run and what their output looks like. <strong>They are
      not findings about anything.</strong>
    </p>
  </div>

  <div class="grid-figures">
    <figure>
      <h3>Calibration</h3>
      {svg_calibration(calib)}
      <figcaption>
        <p>An over-confident model: predictions sit below the diagonal at the top end,
        so it claims more risk than occurs. ECE
        <code>{calib["ece"]}</code>, Brier <code>{calib["brier"]}</code>.</p>
        <p class="fixture">Fixture: {esc(calib["fixture"])}</p>
      </figcaption>
    </figure>

    <figure>
      <h3>Qini</h3>
      {svg_qini(qini)}
      <figcaption>
        <p>The ranking finds persuadable people early, then flattens onto the random
        line once they are exhausted. Coefficient
        <code>{qini["coefficient"]}</code> against
        <code>{qini["total_incremental"]}</code> total incremental responders.</p>
        <p class="fixture">Fixture: {esc(qini["fixture"])}</p>
      </figcaption>
    </figure>

    <figure>
      <h3>Pinball loss by quantile</h3>
      {svg_pinball(pinball)}
      <figcaption>
        <p>Two forecasters with the same centre. The over-confident one loses most in
        the tails &mdash; exactly where a high service level reads the distribution.</p>
        <p class="fixture">Fixture: {esc(pinball["fixture"])}</p>
      </figcaption>
    </figure>

    <figure>
      <h3>Expected cost by threshold</h3>
      {svg_cost_curve(cost)}
      <figcaption>
        <p>Empirical optimum <code>{cost["optimal_threshold"]}</code> against the
        analytic <code>{cost["analytic_threshold"]}</code>. The basin is flat, so the
        cost estimate deserves more scrutiny than the cutoff.</p>
        <p class="fixture">Fixture: {esc(cost["fixture"])}</p>
      </figcaption>
    </figure>
  </div>
</section>

<section class="wrap api">
  <div class="section-head">
    <h2>API</h2>
    <p class="section-note">
      Signatures and summaries are read from the installed package. Each worked example
      is executed as a doctest by the test suite, so an example that drifts from its
      code fails the build.
    </p>
  </div>
  {"".join(api_html)}
</section>
</main>

<footer class="wrap">
  <h2>Provenance</h2>
  <p>
    Every number above comes from <code>reports/proof.json</code>, written by
    <code>scripts/proof.py</code> and rendered by <code>docs/build.py</code>. Nothing
    on this page is typed by hand. The proof file carries no timestamp, so regenerating
    it on unchanged code produces a byte-identical file &mdash; which is what makes
    &ldquo;the page matches the code&rdquo; checkable rather than asserted.
  </p>
  <pre class="example"><code>make all   # lint, test, proof, docs</code></pre>
  <ul class="versions">
    <li><span>spine</span><span>{esc(proof["spine_version"])}</span></li>
    <li><span>python</span><span>{esc(proof["python"])}</span></li>
    {packages}
    <li><span>seed</span><span>{proof["seed"]}</span></li>
  </ul>
  <p class="colophon">
    Krishna Mathur &middot; Master of AI in Business, Term 4 &middot; academic project.
    Set in Spectral and IBM Plex Mono.
  </p>
</footer>

<script>{SCRIPT}</script>
</body>
</html>
"""


STYLE = """
:root {
  --ground: #FBFBF9;
  --surface: #FFFFFF;
  --ink: #14161A;
  --ink-muted: #5B6068;
  --rule: #DCDDD8;
  --rule-soft: #EDEEE9;
  --accent: #2B4C9B;
  --accent-soft: #E4E9F5;
  --graphite: #9DA3AC;
  --fail: #A02B1F;
  --font-prose: "Spectral", "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --font-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --measure: 34rem;
  --step: 1.5rem;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #101215;
    --surface: #171A1E;
    --ink: #E6E7E4;
    --ink-muted: #8B9096;
    --rule: #262A2F;
    --rule-soft: #1D2126;
    --accent: #7A9BE8;
    --accent-soft: #1B2434;
    --graphite: #656B74;
    --fail: #E07A6E;
  }
}
:root[data-theme="dark"] {
  --ground: #101215;
  --surface: #171A1E;
  --ink: #E6E7E4;
  --ink-muted: #8B9096;
  --rule: #262A2F;
  --rule-soft: #1D2126;
  --accent: #7A9BE8;
  --accent-soft: #1B2434;
  --graphite: #656B74;
  --fail: #E07A6E;
}

*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--font-prose);
  font-size: 17px;
  line-height: 1.62;
  font-weight: 300;
  font-variant-numeric: tabular-nums;
}
h1, h2, h3 { font-weight: 600; line-height: 1.18; margin: 0; letter-spacing: -0.012em; }
p { margin: 0 0 0.9em; }
code { font-family: var(--font-mono); font-size: 0.86em; font-weight: 500; }
a { color: var(--accent); }

.wrap { width: min(100% - 2.5rem, 68rem); margin-inline: auto; }
.skip {
  position: absolute; left: -9999px; top: 0; background: var(--accent);
  color: var(--ground); padding: 0.6rem 1rem; z-index: 10;
}
.skip:focus { left: 0; }
:where(a, button):focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 3px;
}

/* ---------------------------------------------------------------- masthead */
.masthead { border-bottom: 1px solid var(--rule); background: var(--ground); }
.masthead-inner {
  display: flex; align-items: baseline; justify-content: space-between;
  padding: 1.1rem 0 0.9rem; gap: 1rem;
}
.wordmark { display: flex; align-items: baseline; gap: 0.85rem; flex-wrap: wrap; }
.wordmark-name {
  font-family: var(--font-mono); font-weight: 600; font-size: 1.02rem;
  letter-spacing: 0.24em;
}
.wordmark-meta {
  font-family: var(--font-mono); font-size: 0.72rem; color: var(--ink-muted);
  letter-spacing: 0.08em;
}
.theme {
  font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.08em;
  background: none; border: 1px solid var(--ink-muted); color: var(--ink-muted);
  padding: 0.42rem 0.8rem; cursor: pointer; display: inline-flex; align-items: center;
  gap: 0.5rem; transition: border-color 150ms ease, color 150ms ease;
  min-height: 34px;
}
.theme:hover { border-color: var(--accent); color: var(--ink); }
.theme-dot {
  width: 8px; height: 8px; border: 1px solid currentColor; border-radius: 50%;
  background: linear-gradient(90deg, currentColor 50%, transparent 50%);
}

/* -------------------------------------------------------------------- hero */
.hero { padding: 4.5rem 0 3rem; }
.eyebrow {
  font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--ink-muted); margin-bottom: 1.6rem;
}
h1 { font-size: clamp(1.85rem, 1.1rem + 3.1vw, 3.35rem); max-width: 20ch; }
h1 + .lede { margin-top: 1.5rem; }
.lede { max-width: var(--measure); font-size: 1.06rem; color: var(--ink-muted); }

.hero-figure { margin: 3rem 0 0; }
.fig { width: 100%; height: auto; display: block; overflow: visible; }
.fig-hero { max-width: 52rem; }
figcaption { color: var(--ink-muted); font-size: 0.88rem; }
.hero-figure figcaption { margin-top: 1rem; max-width: var(--measure); }
.key {
  display: flex; gap: 1.4rem; flex-wrap: wrap; font-family: var(--font-mono);
  font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase;
  margin-bottom: 0.9rem;
}
.key span { display: inline-flex; align-items: center; gap: 0.45rem; }
.swatch { width: 15px; height: 9px; display: inline-block; }
.swatch-train { background: var(--graphite); }
.swatch-test { background: var(--accent); }
.swatch-gap {
  background: repeating-linear-gradient(
    45deg, var(--rule), var(--rule) 2px, transparent 2px, transparent 4px
  );
  border: 1px solid var(--rule);
}

.assertion {
  margin-top: 2.4rem; padding-top: 1.4rem; border-top: 1px solid var(--rule);
  font-family: var(--font-mono); font-size: 0.82rem; line-height: 1.9;
  color: var(--ink-muted); max-width: 46rem;
}
.assertion strong { color: var(--ink); font-weight: 600; }
.assertion-rule { display: block; margin-top: 0.5rem; }
.assertion-rule code { color: var(--accent); }

/* ------------------------------------------------------------------- scope */
.scope {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr));
  gap: 2.5rem; padding: 3rem 0; border-top: 1px solid var(--rule);
}
.scope h2 {
  font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--ink-muted); margin-bottom: 1.1rem;
}
.scope ul { list-style: none; margin: 0; padding: 0; }
.scope li {
  padding: 0.7rem 0; border-top: 1px solid var(--rule-soft); font-size: 0.95rem;
}
.scope-col-negative li { color: var(--ink-muted); }

/* ------------------------------------------------------------ section heads */
.section-head { padding: 3rem 0 1.8rem; border-top: 1px solid var(--rule); }
.section-head h2 { font-size: clamp(1.35rem, 1rem + 1.2vw, 1.85rem); }
.section-note { max-width: var(--measure); color: var(--ink-muted); margin-top: 0.7rem; }

/* ------------------------------------------------------------------ ledger */
.ledger { border-top: 1px solid var(--rule); }
.ledger-row {
  display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 1.5rem 2.5rem;
  align-items: start; padding: 1.15rem 0; border-bottom: 1px solid var(--rule-soft);
}
.ledger-row.failed { background: color-mix(in srgb, var(--fail) 7%, transparent); }
.ledger-claim p { margin: 0; font-size: 0.97rem; }
.ledger-source {
  font-family: var(--font-mono); font-size: 0.71rem; color: var(--ink-muted);
  margin-top: 0.4rem !important; letter-spacing: 0.01em; word-break: break-word;
}
.ledger-source .sep { opacity: 0.62; font-style: normal; }

.ledger-figs { display: flex; gap: 1.5rem; align-items: baseline; }
.cell { display: flex; flex-direction: column; gap: 0.28rem; }
.cell-key {
  font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.13em;
  text-transform: uppercase; color: var(--ink-muted);
}
.num {
  font-family: var(--font-mono); font-size: 0.86rem; font-weight: 500;
  display: inline-flex; white-space: nowrap;
}
.num-w { flex: 1 1 auto; text-align: right; min-width: 3.1ch; }
.num-p { flex: 0 0 auto; opacity: 0.5; }
.num-f { flex: 1 1 auto; text-align: left; min-width: 6ch; }
.num-sci { justify-content: flex-end; min-width: 8ch; color: var(--ink-muted); }
.cell-delta .num-sci { color: var(--ink); }
.mark {
  font-family: var(--font-mono); font-size: 0.63rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--accent); border: 1px solid var(--accent);
  padding: 0.24rem 0.5rem; white-space: nowrap;
}
.failed .mark { color: var(--fail); border-color: var(--fail); }

/* ----------------------------------------------------------------- figures */
.grid-figures {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 26rem), 1fr));
  gap: 3rem 3.5rem;
}
.grid-figures figure { margin: 0; }
.grid-figures h3 {
  font-family: var(--font-mono); font-size: 0.74rem; letter-spacing: 0.12em;
  text-transform: uppercase; padding-bottom: 0.8rem; margin-bottom: 1.2rem;
  border-bottom: 1px solid var(--rule);
}
.grid-figures figcaption { margin-top: 0.9rem; }
.fixture {
  font-family: var(--font-mono); font-size: 0.7rem; color: var(--ink-muted);
  border-left: 2px solid var(--rule); padding-left: 0.7rem; margin-top: 0.7rem;
}

/* --------------------------------------------------------------------- svg */
.bar-train { fill: var(--graphite); }
.bar-test { fill: var(--accent); }
.bar-gap { fill: var(--rule); opacity: 0.85; }
.origin { stroke: var(--accent); stroke-width: 1; }
.axis { stroke: var(--rule); stroke-width: 1; }
.grid { stroke: var(--rule-soft); stroke-width: 1; }
.reference { stroke: var(--rule); stroke-width: 1; stroke-dasharray: 3 3; }
.series { fill: none; stroke: var(--accent); stroke-width: 1.6; }
.series-muted { fill: none; stroke: var(--graphite); stroke-width: 1.4; }
.point { fill: var(--accent); }
.point-muted { fill: var(--graphite); }
.svg-label-r, .svg-tick, .svg-note {
  font-family: var(--font-mono); fill: var(--ink-muted); font-size: 9.5px;
}
.svg-label-r { text-anchor: end; }
.svg-tick { text-anchor: middle; }
.svg-tick-r { text-anchor: end; }
.svg-note { text-anchor: start; font-size: 9px; }
.svg-note-end { text-anchor: end; }
.svg-note-accent { fill: var(--accent); }

.fold { opacity: 0; animation: reveal 220ms ease-out forwards; animation-delay: var(--d); }
@keyframes reveal {
  from { opacity: 0; transform: translateX(-6px); }
  to { opacity: 1; transform: none; }
}

/* --------------------------------------------------------------------- api */
.api-module { padding-top: 2.2rem; }
.api-module > h2 { font-size: 1.05rem; font-family: var(--font-mono); font-weight: 600; }
.module-blurb { color: var(--ink-muted); font-size: 0.92rem; margin-top: 0.4rem; }
.member { padding: 1.4rem 0; border-top: 1px solid var(--rule-soft); }
.member h3 {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.55rem;
  font-weight: 400; font-size: 0.95rem;
}
.member-kind {
  font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.13em;
  text-transform: uppercase; color: var(--ink-muted); border: 1px solid var(--rule);
  padding: 0.14rem 0.4rem;
}
.member h3 code { font-size: 0.95rem; font-weight: 600; color: var(--accent); }
.member-sig {
  font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink-muted);
  word-break: break-word;
}
.member p { margin: 0.55rem 0 0; max-width: var(--measure); font-size: 0.95rem; }
.example {
  font-family: var(--font-mono); font-size: 0.76rem; line-height: 1.75;
  background: var(--surface); border: 1px solid var(--rule); padding: 0.9rem 1.1rem;
  margin: 0.9rem 0 0; overflow-x: auto; white-space: pre;
}
.example code { font-size: inherit; font-weight: 400; }

/* ------------------------------------------------------------------ footer */
footer { padding: 3rem 0 5rem; border-top: 1px solid var(--rule); margin-top: 3rem; }
footer h2 {
  font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--ink-muted); margin-bottom: 1.1rem;
}
footer p { max-width: var(--measure); color: var(--ink-muted); font-size: 0.92rem; }
.versions {
  list-style: none; padding: 0; margin: 1.8rem 0 0; display: grid;
  grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: 0;
  border-top: 1px solid var(--rule);
}
.versions li {
  display: flex; justify-content: space-between; gap: 1rem; padding: 0.55rem 0;
  border-bottom: 1px solid var(--rule-soft); font-family: var(--font-mono);
  font-size: 0.73rem; margin-right: 2rem;
}
.versions li span:first-child { color: var(--ink-muted); }
.colophon { margin-top: 2rem; font-size: 0.82rem; }

/* ------------------------------------------------------------- narrow view */
@media (max-width: 720px) {
  body { font-size: 16px; }
  .hero { padding: 3rem 0 2rem; }
  .ledger-row { grid-template-columns: 1fr; gap: 1rem; }
  .ledger-figs { flex-wrap: wrap; gap: 1rem 1.6rem; }
  .grid-figures { gap: 2.5rem; }
}

@media (prefers-reduced-motion: reduce) {
  .fold { opacity: 1; animation: none; }
  * { transition-duration: 1ms !important; }
}
"""

SCRIPT = """
(function () {
  var root = document.documentElement;
  var button = document.getElementById('theme');
  var label = document.getElementById('theme-label');

  function systemDark() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function current() {
    var set = root.getAttribute('data-theme');
    return set || (systemDark() ? 'dark' : 'light');
  }
  function paint() {
    var next = current() === 'dark' ? 'Light' : 'Dark';
    label.textContent = next;
    button.setAttribute('aria-label', 'Switch to ' + next.toLowerCase() + ' theme');
  }
  try {
    var saved = localStorage.getItem('spine-theme');
    if (saved === 'dark' || saved === 'light') root.setAttribute('data-theme', saved);
  } catch (e) { /* private browsing, or storage blocked */ }

  paint();
  button.addEventListener('click', function () {
    var next = current() === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('spine-theme', next); } catch (e) { /* ignore */ }
    paint();
  });
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', paint);
  }
})();
"""


def main() -> int:
    """Render the page.

    Returns
    -------
    int
        Process exit code.
    """
    if not PROOF.exists():
        raise SystemExit(
            f"{PROOF.relative_to(ROOT)} is missing; run `uv run python scripts/proof.py` first"
        )
    proof = json.loads(PROOF.read_text())
    OUTPUT.write_text(render(proof))
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
