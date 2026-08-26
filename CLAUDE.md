# CLAUDE.md

Drop this at the root of every Term 4 repo. Adjust the "This repo" section per project.

---

## Context

Academic project for the Master of AI in Business (MAIB), SP Jain School of Global Management, Dubai — Term 4. Author: Krishna Mathur.

Work here is submitted for grading and viva examination, and forms part of a public portfolio. That means two things: every claim must be defensible under questioning, and nothing may be overstated.

## This repo

<!-- Replace per project -->
**Name:** SPINE / ADIL / ATHAR / QARAR / MIZAN
**Subject:** MAIB AI ___
**One-line purpose:** ___

---

## How to work with me

### Think before coding
- State assumptions explicitly before implementing. If uncertain, ask.
- If multiple interpretations exist, present them. Don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop and name what's confusing.

### Simplicity first
- Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No configurability that wasn't requested.
- If you write 200 lines and it could be 50, rewrite it.

Test: would a senior engineer call this overcomplicated? If yes, simplify.

### Surgical changes
- Touch only what you must.
- Don't "improve" adjacent code, comments or formatting.
- Match existing style even if you'd do it differently.
- Notice unrelated dead code? Mention it, don't delete it.
- Remove only the imports and variables *your* change orphaned.

Every changed line should trace to an explicit request.

### Goal-driven execution
Turn every task into a verifiable goal before starting:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

"Make it work" is not a success criterion. "pytest passes and the
newsvendor matches the hand-computed value to 6dp" is.

---

## Academic integrity rules

These are non-negotiable and override convenience.

1. **Never invent numbers.** If a result isn't computed, say it isn't computed. No illustrative figures presented as findings.
2. **Never overstate.** "The model achieved X on this holdout" — not "the model achieves X."
3. **Synthetic data must be labelled at every output.** If a dataset is synthetic or a spend panel is simulated, every chart, table and paragraph derived from it carries the caveat.
4. **Proxy attributes are not protected attributes.** Fairness analysis on proxies is a methodological demonstration. Label it as such everywhere.
5. **Negative results are results.** If the challenger model loses, or the hypothesis is falsified, report it plainly. Do not reframe to rescue a narrative.
6. **Cite regulatory text by clause.** Any compliance claim names its source provision. No paraphrased regulation without provenance.
7. **Nothing here is legal advice.** Governance work is my reading of published guidance, not a lawyer's.

---

## Code conventions

| Item | Convention |
|---|---|
| Python | 3.11 |
| Env | `uv` |
| Format/lint | `ruff` |
| Tests | `pytest`, in `tests/`, mirroring `src/` layout |
| Notebooks | Numbered `NN_name.ipynb`, outputs cleared before commit |
| Data | `data/raw/` (never committed), `data/processed/` (never committed) |
| Reports | `reports/` — markdown, committed |
| Secrets | `.env`, never committed; `.env.example` documents required keys |

- Docstrings on every public function: what it does, the formula if it has one, one worked example.
- Type hints on public functions.
- No `print()` in library code — use `logging`.
- Random seeds fixed and recorded in every experiment.

## Evaluation conventions

- **Never random-split temporal data.** Use `spine.splitting`.
- **Never MAPE on series containing zeros.** Use MASE.
- Report intervals, not just point estimates, wherever a model produces a posterior or a quantile.
- Every metric reported alongside the split it was computed on.
- Baselines must be strong. A weak baseline is a form of dishonesty.

---

## What "done" means

A session is done when:
1. Its stated verification check passes
2. `ruff check` is clean
3. `pytest` is green
4. The results are recorded somewhere durable (a report, a parquet, a table) — not just printed in a cell

A project is done when someone else can clone it, run `make setup && make all`, and reproduce every number in the report.

---

## Things I will ask you in the viva

Write code that survives these:

- Why this metric and not the obvious alternative?
- What's your baseline and is it strong enough?
- Where does this leak information from the future?
- What would falsify your conclusion?
- Which assumption, if wrong, breaks everything?
- What did you choose *not* to do, and why?

If a piece of code can't survive one of these, it isn't finished.
