"""Run every SPINE verification and record the results in reports/proof.json.

The docs page is built from this file. Nothing on that page is typed by a human,
which is the point: a claim about a number and the number itself cannot drift
apart if one is generated from the other.

Deliberately contains no timestamp. Re-running this script on an unchanged
library must produce a byte-identical file, so that `git diff` proves the page
matches the code. Provenance comes from the package versions and the seed
recorded below, and from git history for the when.

Run: uv run python scripts/proof.py
"""

from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import version
from itertools import product
from pathlib import Path
from statistics import NormalDist

import numpy as np
from scipy import stats

import spine
from spine.decisions import (
    critical_fractile,
    expected_cost_curve,
    newsvendor_cost,
    newsvendor_empirical,
    newsvendor_parametric,
    optimal_threshold,
    safety_stock,
)
from spine.metrics import (
    brier_score,
    calibration_bins,
    expected_calibration_error,
    mase,
    mean_pinball_by_quantile,
    pr_auc,
    qini_auc,
    qini_curve,
)
from spine.splitting import RollingOriginSplit

SEED = 20260827
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "proof.json"


def check(
    identifier: str,
    claim: str,
    source: str,
    computed: float,
    expected: float,
    expected_from: str,
    tolerance: float,
) -> dict:
    """Record one verification as a ledger row.

    Parameters
    ----------
    identifier : str
        Stable key for the check.
    claim : str
        What is being asserted, in words.
    source : str
        Fully qualified name of the function under test.
    computed : float
        What SPINE produced.
    expected : float
        What it should be, from an independent source.
    expected_from : str
        Where the expected value came from.
    tolerance : float
        Absolute tolerance the delta must fall inside.

    Returns
    -------
    dict
        The ledger row.
    """
    delta = abs(computed - expected)
    return {
        "id": identifier,
        "claim": claim,
        "source": source,
        "computed": computed,
        "expected": expected,
        "expected_from": expected_from,
        "delta": delta,
        "tolerance": tolerance,
        "passed": bool(delta <= tolerance),
    }


def numerical_verifications() -> list[dict]:
    """Cross-check SPINE against independent implementations and hand arithmetic.

    Returns
    -------
    list of dict
        Ledger rows.
    """
    rows = []

    # --- newsvendor, against the standard library's normal distribution --------
    demand = stats.norm(loc=100, scale=20)
    rows.append(
        check(
            "newsvendor-parametric",
            "Optimal order for demand ~ Normal(100, 20) with cu=5, co=2",
            "spine.decisions.newsvendor_parametric",
            computed=newsvendor_parametric(demand, cu=5.0, co=2.0),
            expected=NormalDist(100, 20).inv_cdf(5 / 7),
            expected_from="statistics.NormalDist, Python standard library",
            tolerance=1e-9,
        )
    )
    rows.append(
        check(
            "newsvendor-optimality",
            "That order quantity sits exactly at the critical fractile: F(Q*) = CF",
            "spine.decisions.newsvendor_parametric",
            computed=float(demand.cdf(newsvendor_parametric(demand, cu=5.0, co=2.0))),
            expected=critical_fractile(5.0, 2.0),
            expected_from="the newsvendor optimality condition itself",
            tolerance=1e-12,
        )
    )
    rows.append(
        check(
            "critical-fractile",
            "Critical fractile for cu=5, co=2 is 5/7",
            "spine.decisions.critical_fractile",
            computed=critical_fractile(5.0, 2.0),
            expected=5 / 7,
            expected_from="hand arithmetic: cu / (cu + co)",
            tolerance=1e-15,
        )
    )

    # --- empirical newsvendor recovers the analytic one on a dense grid --------
    levels = np.linspace(0.001, 0.999, 999)
    rows.append(
        check(
            "newsvendor-empirical",
            "Order read off a 999-point quantile grid recovers the analytic optimum",
            "spine.decisions.newsvendor_empirical",
            computed=newsvendor_empirical(demand.ppf(levels), levels, cu=5.0, co=2.0),
            expected=newsvendor_parametric(demand, cu=5.0, co=2.0),
            expected_from="spine.decisions.newsvendor_parametric on the same distribution",
            tolerance=0.01,
        )
    )

    # --- safety stock, against the standard library ---------------------------
    rows.append(
        check(
            "safety-stock",
            "Safety stock at a 95% service level, demand sd 20 over a 4-period lead time",
            "spine.decisions.safety_stock",
            computed=safety_stock(demand_std=20.0, lead_time=4.0, service_level=0.95),
            expected=NormalDist().inv_cdf(0.95) * 20 * 2,
            expected_from="statistics.NormalDist, Python standard library",
            tolerance=1e-9,
        )
    )

    # --- MASE, against utilsforecast ------------------------------------------
    import pandas as pd
    from utilsforecast.losses import mase as uf_mase

    rng = np.random.default_rng(SEED)
    seasonality, n_train, n_test = 7, 120, 28
    t = np.arange(n_train + n_test)
    series = 100 + 10 * np.sin(2 * np.pi * t / seasonality) + rng.normal(0, 3, size=t.size)
    y_train, y_true = series[:n_train], series[n_train:]
    y_pred = y_true + rng.normal(0, 2, size=n_test)

    reference = float(
        uf_mase(
            pd.DataFrame(
                {
                    "unique_id": "s1",
                    "ds": pd.RangeIndex(n_train, n_train + n_test),
                    "y": y_true,
                    "model": y_pred,
                }
            ),
            models=["model"],
            seasonality=seasonality,
            train_df=pd.DataFrame({"unique_id": "s1", "ds": pd.RangeIndex(n_train), "y": y_train}),
        )["model"].iloc[0]
    )
    rows.append(
        check(
            "mase-utilsforecast",
            "MASE on a seasonal reference series matches an independent implementation",
            "spine.metrics.mase",
            computed=mase(y_true, y_pred, y_train, seasonality=seasonality),
            expected=reference,
            expected_from="utilsforecast.losses.mase (Nixtla)",
            tolerance=1e-12,
        )
    )

    # --- MASE of a naive forecast on a random walk is 1 by construction -------
    walk = np.cumsum(np.random.default_rng(SEED + 1).normal(0, 1, size=4000))
    walk_train, walk_test = walk[:3000], walk[3000:]
    naive = np.concatenate([[walk_train[-1]], walk_test[:-1]])
    rows.append(
        check(
            "mase-naive-is-one",
            "A naive forecast on a random walk scores 1.0: no better than naive",
            "spine.metrics.mase",
            computed=mase(walk_test, naive, walk_train, seasonality=1),
            expected=1.0,
            expected_from="the definition of MASE",
            tolerance=0.05,
        )
    )

    # --- PR-AUC, against scikit-learn -----------------------------------------
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(SEED + 2)
    labels = rng.binomial(1, 0.08, size=5000)
    scores = rng.uniform(size=5000) + 0.4 * labels
    rows.append(
        check(
            "pr-auc-sklearn",
            "PR-AUC on an imbalanced sample matches scikit-learn's average precision",
            "spine.metrics.pr_auc",
            computed=pr_auc(labels, scores),
            expected=float(average_precision_score(labels, scores)),
            expected_from="sklearn.metrics.average_precision_score",
            tolerance=1e-12,
        )
    )

    # --- calibration: a calibrated model has ECE 0 ----------------------------
    calibrated_prob = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
    calibrated_true = np.concatenate([np.repeat([1, 0], [20, 80]), np.repeat([1, 0], [80, 20])])
    rows.append(
        check(
            "ece-calibrated",
            "A perfectly calibrated sample has zero expected calibration error",
            "spine.metrics.expected_calibration_error",
            computed=expected_calibration_error(calibrated_true, calibrated_prob, n_bins=10),
            expected=0.0,
            expected_from="the definition of calibration",
            tolerance=1e-12,
        )
    )
    rows.append(
        check(
            "brier-hand",
            "Brier score of [0.1, 0.2, 0.3, 0.9] against [0, 0, 1, 1] is 0.55/4",
            "spine.metrics.brier_score",
            computed=brier_score([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9]),
            expected=0.55 / 4,
            expected_from="hand arithmetic: (0.01 + 0.04 + 0.49 + 0.01) / 4",
            tolerance=1e-12,
        )
    )

    # --- cost-sensitive threshold converges on the analytic optimum -----------
    rng = np.random.default_rng(SEED + 3)
    probability = rng.uniform(size=200_000)
    outcome = rng.binomial(1, probability)
    costs = np.array([[0.0, 1.0], [5.0, 0.0]])
    threshold, cost_at_empirical = optimal_threshold(outcome, probability, costs)
    rows.append(
        check(
            "threshold-analytic",
            "For calibrated probabilities the cost-optimal cutoff is C_FP / (C_FP + C_FN)",
            "spine.decisions.optimal_threshold",
            computed=threshold,
            expected=1 / 6,
            # The empirical argmin is a noisy statistic: the cost basin is flat
            # near the optimum, so many nearby thresholds are nearly as good and
            # the minimum wanders between them from sample to sample. The next
            # check measures that flatness rather than assuming it away.
            expected_from="hand arithmetic: 1 / (1 + 5)",
            tolerance=0.02,
        )
    )

    # Flatness is a relative claim, so measure it relatively: how much more does
    # the analytic cutoff cost than the empirical optimum, in percent?
    curve_at = expected_cost_curve(outcome, probability, costs)
    nearest_analytic = float(
        curve_at.iloc[(curve_at["threshold"] - 1 / 6).abs().idxmin()]["expected_cost"]
    )
    penalty_percent = 100.0 * (nearest_analytic - cost_at_empirical) / nearest_analytic
    rows.append(
        check(
            "threshold-basin-is-flat",
            "Using the analytic cutoff instead of the empirical optimum costs under "
            "0.5% more, so the exact cutoff matters far less than the cost estimate",
            "spine.decisions.expected_cost_curve",
            computed=penalty_percent,
            expected=0.0,
            expected_from="percentage cost penalty; 0 would mean the two cutoffs tie",
            tolerance=0.5,
        )
    )

    # --- simulation: the analytic order really is cheapest --------------------
    rng = np.random.default_rng(SEED + 4)
    realised = rng.normal(100, 20, size=200_000)
    optimal_order = newsvendor_parametric(demand, cu=5.0, co=2.0)

    def mean_cost(order: float) -> float:
        return float(newsvendor_cost(np.full_like(realised, order), realised, 5.0, 2.0).mean())

    ordering_the_mean = mean_cost(100.0)
    at_optimum = mean_cost(optimal_order)
    rows.append(
        check(
            "newsvendor-simulation",
            "Simulated over 200,000 periods, ordering to the critical fractile beats "
            "ordering the mean forecast",
            "spine.decisions.newsvendor_cost",
            computed=at_optimum,
            expected=ordering_the_mean,
            expected_from="mean realised cost of ordering the mean, on the same demand draw",
            tolerance=float("inf"),
        )
    )
    rows[-1]["passed"] = bool(at_optimum < ordering_the_mean)
    rows[-1]["comparison"] = "lower is better"
    rows[-1]["saving_percent"] = 100.0 * (ordering_the_mean - at_optimum) / ordering_the_mean

    return rows


def leak_safety() -> dict:
    """Exhaustively check the split boundary over a deterministic grid.

    The test suite also checks this property on 250 randomised cases generated by
    hypothesis. This grid is the deterministic counterpart: same property, every
    combination, reproducible byte for byte.

    Returns
    -------
    dict
        Case count, violation count and the grid that was searched.
    """
    cases = violations = 0
    grid = {
        "n_samples": [40, 80, 150],
        "n_splits": [1, 2, 3, 4, 5],
        "horizon": [1, 2, 3, 4, 5, 6],
        "gap": [0, 1, 2, 3, 4],
        "step": [1, 2, 3, 4],
        "window": ["expanding", "sliding-5", "sliding-10", "sliding-20"],
    }

    for n_samples, n_splits, horizon, gap, step, window in product(*grid.values()):
        sliding = window.startswith("sliding")
        try:
            splitter = RollingOriginSplit(
                n_splits=n_splits,
                horizon=horizon,
                step=step,
                gap=gap,
                window="sliding" if sliding else "expanding",
                window_size=int(window.split("-")[1]) if sliding else None,
            )
            folds = list(splitter.split(np.arange(n_samples)))
        except ValueError:
            continue  # the configuration does not fit the series; refusing is correct

        for train, test in folds:
            cases += 1
            if not (train.max() + gap < test.min()):
                violations += 1
            if set(train.tolist()) & set(test.tolist()):
                violations += 1

    return {
        "cases": cases,
        "violations": violations,
        "grid": {key: [str(v) for v in values] for key, values in grid.items()},
        "assertion": "max(train_index) + gap < min(test_index), for every fold",
        "hypothesis_cases": 250,
    }


def figures() -> dict:
    """Produce the arrays the docs page draws.

    Every array here comes from a seeded synthetic fixture, not from a dataset.
    The page labels them as such: they demonstrate that the functions run and
    what their output looks like. They are not findings about anything.

    Returns
    -------
    dict
        Named figure payloads.
    """
    rng = np.random.default_rng(SEED)

    # --- the hero: an actual rolling-origin layout ----------------------------
    n_samples, gap = 60, 2
    splitter = RollingOriginSplit(n_splits=5, horizon=6, step=6, gap=gap)
    folds = [
        {
            "train_start": int(train.min()),
            "train_end": int(train.max()) + 1,
            "test_start": int(test.min()),
            "test_end": int(test.max()) + 1,
        }
        for train, test in splitter.split(np.arange(n_samples))
    ]

    # --- calibration of a deliberately over-confident fixture model -----------
    truth_probability = rng.uniform(0.05, 0.95, size=4000)
    observed = rng.binomial(1, truth_probability)
    # Push predictions towards the extremes: the classic over-confident model.
    overconfident = np.clip(truth_probability + 0.25 * (truth_probability - 0.5), 0.01, 0.99)
    bins = calibration_bins(observed, overconfident, n_bins=10, strategy="uniform")

    # --- Qini, on a fixture with a real responder subgroup --------------------
    n_uplift = 6000
    treatment = rng.binomial(1, 0.5, size=n_uplift)
    persuadable = rng.binomial(1, 0.35, size=n_uplift)
    uplift_outcome = np.where(
        treatment == 1,
        np.maximum(persuadable, rng.binomial(1, 0.05, size=n_uplift)),
        rng.binomial(1, 0.05, size=n_uplift),
    )
    signal = persuadable + rng.normal(0, 0.6, size=n_uplift)
    fraction, gain = qini_curve(uplift_outcome, treatment, signal)
    step = max(1, len(fraction) // 200)

    # --- pinball loss across quantiles ---------------------------------------
    quantile_levels = np.round(np.arange(0.1, 0.95, 0.1), 2)
    actual = rng.normal(100, 20, size=3000)
    sharp = np.column_stack([np.full(3000, stats.norm(100, 20).ppf(q)) for q in quantile_levels])
    too_narrow = np.column_stack(
        [np.full(3000, stats.norm(100, 8).ppf(q)) for q in quantile_levels]
    )

    # --- expected cost curve --------------------------------------------------
    cost_probability = rng.uniform(size=20_000)
    cost_outcome = rng.binomial(1, cost_probability)
    curve = expected_cost_curve(cost_outcome, cost_probability, np.array([[0.0, 1.0], [5.0, 0.0]]))
    curve = curve.iloc[:: max(1, len(curve) // 300)]
    best_threshold, best_cost = optimal_threshold(
        cost_outcome, cost_probability, np.array([[0.0, 1.0], [5.0, 0.0]])
    )

    return {
        "rolling_origin": {
            "n_samples": n_samples,
            "gap": gap,
            "folds": folds,
            "config": "RollingOriginSplit(n_splits=5, horizon=6, step=6, gap=2)",
        },
        "calibration": {
            "mean_predicted": bins["mean_predicted"].round(6).tolist(),
            "observed_rate": bins["observed_rate"].round(6).tolist(),
            "n": bins["n"].tolist(),
            "ece": round(expected_calibration_error(observed, overconfident, n_bins=10), 6),
            "brier": round(brier_score(observed, overconfident), 6),
            "fixture": "4,000 synthetic cases; an over-confident model, seeded",
        },
        "qini": {
            "fraction": np.round(fraction[::step], 6).tolist(),
            "gain": np.round(gain[::step], 6).tolist(),
            "total_incremental": round(float(gain[-1]), 6),
            "coefficient": round(qini_auc(uplift_outcome, treatment, signal), 6),
            "fixture": "6,000 synthetic cases, randomised treatment, 35% persuadable",
        },
        "pinball": {
            "levels": quantile_levels.tolist(),
            "sharp": np.round(mean_pinball_by_quantile(actual, sharp, quantile_levels), 6).tolist(),
            "too_narrow": np.round(
                mean_pinball_by_quantile(actual, too_narrow, quantile_levels), 6
            ).tolist(),
            "fixture": "3,000 draws from Normal(100, 20); one forecaster with the right "
            "spread, one over-confident at sd 8",
        },
        "cost_curve": {
            "threshold": curve["threshold"].round(6).tolist(),
            "expected_cost": curve["expected_cost"].round(6).tolist(),
            "optimal_threshold": round(best_threshold, 6),
            "optimal_cost": round(best_cost, 6),
            "analytic_threshold": round(1 / 6, 6),
            "fixture": "20,000 synthetic cases with calibrated probabilities; "
            "a false negative costs 5x a false positive",
        },
    }


def main() -> int:
    """Write reports/proof.json and report whether every verification passed.

    Returns
    -------
    int
        Process exit code: 0 when every verification passed.
    """
    verifications = numerical_verifications()
    payload = {
        "generated_by": "scripts/proof.py",
        "note": (
            "No timestamp is recorded, so re-running this script on an unchanged "
            "library produces a byte-identical file. Provenance is the seed and the "
            "package versions below; the when is in git history."
        ),
        "seed": SEED,
        "spine_version": spine.__version__,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "utilsforecast")
        },
        "verifications": verifications,
        "leak_safety": leak_safety(),
        "figures": figures(),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")

    failed = [row["id"] for row in verifications if not row["passed"]]
    leaks = payload["leak_safety"]["violations"]
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"  {len(verifications) - len(failed)}/{len(verifications)} verifications passed")
    print(f"  {payload['leak_safety']['cases']} split folds checked, {leaks} boundary violations")
    if failed:
        print(f"  FAILED: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed or leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
