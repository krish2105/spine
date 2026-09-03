# SPINE v2 — Session Plan (Week 0, 3 sessions)

## Decision: EXTEND the existing SPINE, do not rebuild
It already passes tests with high coverage. Rebuilding throws away working,
tested code to gain nothing. Extend it — and history in a repo is an asset,
not clutter.

## Decision: build modules LAZILY
Do NOT build all seven modules in Week 0. Building `spine.eval` before any
consumer exists means designing an interface for an imaginary caller, and
you will rewrite it in Week 4 anyway.

| Module | Built in | Because |
|---|---|---|
| config, errors, db, auth, obs, deploy | **Week 0** | MASAR Session 1 needs all of them |
| spine.ui | **Week 0 (tokens only)**, components as needed | Design tokens must be fixed early; components accrete |
| spine.llm | **MASAR Week 3** | First real LLM consumer is the dispatcher agent |
| spine.eval | **RASID Week 4** | First consumer with serious eval needs |

Rule: a module graduates into spine the second time an app needs it. First
use lives in the app. Second use triggers extraction. This prevents the
classic failure of a shared library full of abstractions nobody uses.

---
## Session 1 — Audit and extend the foundation
Attach: `@docs/ARCHITECTURE.md @docs/MODULE-CONTRACTS.md`
Scope: audit what exists, reconcile against MODULE-CONTRACTS.md, upgrade to
the frozen stack (Python 3.12, SQLAlchemy 2.0 async, Pydantic v2), add
`spine.config.Settings` and `spine.errors`.
Ask Claude Code to report the gap between existing code and the contracts
BEFORE changing anything.
Done: existing tests still pass, gap report written to docs/SESSION-LOG.md.

## Session 2 — db + auth + obs
Attach: `@docs/MODULE-CONTRACTS.md`
Scope: async session factory, `Repository[T]`, Alembic with PostGIS and
pgvector migrations, Supabase JWT verification, `get_current_user`,
`require_role`, structlog config, `@traced`, cost counters.
Done: a protected endpoint returns 401 without a token, 200 with one.
Coverage 85%+ on db and auth.

## Session 3 — Deploy templates + hello world
Attach: `@docs/ARCHITECTURE.md`
Scope: Dockerfile, render.yaml, vercel.json, GitHub Actions CI, UI design
tokens (dark/light, RTL), and a hello-world app deployed end to end.
Done: **a live URL loads with a logged-in user.**

Do not start MASAR until that URL exists. Supabase and Render are already
set up, so this should be a half-day, not a week.
