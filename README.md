# SPINE

**Subject:** shared library
**Purpose:** Evaluation, temporal splitting and decision logic shared by all four downstream projects.

MAIB Term 4 · SP Jain School of Global Management, Dubai · Krishna Mathur

SPINE contains **no models**. It contains the parts that are easy to get subtly wrong and
expensive to get wrong three times: the MASE denominator, the split boundary, the critical
fractile.

## Modules

| Module | Holds |
|---|---|
| `spine.splitting` | Rolling-origin (expanding, sliding), group-aware temporal splits, single-cutoff split |
| `spine.metrics` | MASE, pinball loss, PR-AUC, Brier, expected calibration error, Qini |
| `spine.decisions` | Critical fractile, newsvendor (parametric and empirical), safety stock, cost-sensitive threshold |
| `spine.fairness` | Group rates, disparity ratio, calibration by group |
| `spine.cards` | Model card generator — YAML in, markdown out |
| `spine.io` | Shared `DATA_ROOT`, schema declaration, validating reader |

Two things enforced in code rather than by habit: `mase()` takes the training window as a
required argument, so the leaking denominator is awkward to write by accident; and
`cards.render_card()` refuses to emit a metric without the split it was computed on, or a
fairness section that does not declare whether its attribute is a proxy.

## Setup

```bash
uv sync
cp .env.example .env
```

## Run

```bash
make all      # lint, test, regenerate the proof file and the docs page
```

Individually: `make lint`, `make test`, `make proof`, `make docs`.

## Using SPINE from another repo

```toml
# in the downstream repo's pyproject.toml
[project]
dependencies = ["spine"]

[tool.uv.sources]
spine = { path = "../01-spine", editable = true }
```

Who uses what: **QARAR** → `splitting`, `metrics.mase`/`pinball`, `decisions.newsvendor_*`.
**ADIL** → `splitting`, `metrics` (PR-AUC, Brier, ECE), `fairness`, `decisions.optimal_threshold`,
`cards`. **ATHAR** → `metrics.qini_*`, `splitting`. **MIZAN** → `cards` as the base renderer
beneath BAYAN's clause mapping, and `fairness`.

## Verification

`reports/proof.json` records every verification: what SPINE computed, what an independent
source expected, and the difference. Independent means the Python standard library
(`statistics.NormalDist`), scikit-learn, Nixtla's utilsforecast, or arithmetic done by hand —
never SPINE checking itself.

`docs/index.html` is generated from that file plus the library's own docstrings. No number on
that page is typed by a human. The proof file carries no timestamp, so regenerating it on
unchanged code produces a byte-identical file — which makes "the page matches the code"
checkable rather than asserted:

```bash
rm reports/proof.json docs/index.html && make all && git diff --stat
```

Every worked example in every docstring runs as a doctest under `pytest`, so an example that
drifts from its code fails the suite.

## Limitations

- The newsvendor model is single-period: no lead time, no lot sizing, no substitution.
- `safety_stock` assumes independent demand across periods and a deterministic lead time.
  Both are usually false, and lead-time variability normally dominates in practice.
- Qini requires randomised treatment assignment; on observational data the control
  rescaling has no causal interpretation.
- Figures on the docs page are drawn from seeded synthetic fixtures. They show that the
  functions run. They are not findings.
