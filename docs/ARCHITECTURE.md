# SPINE Architecture

## Why this exists
Five apps in twelve weeks is only possible if auth, database access, LLM
routing, evaluation, observability, UI and deploy config are written once.
Break the reuse chain and the timeline fails.

## Layout
```
spine/
├── src/spine/
│   ├── config.py        Settings (pydantic-settings), the only env reader
│   ├── errors.py        Typed exception hierarchy
│   ├── auth/            Supabase JWT verification, FastAPI dependencies
│   ├── db/              Base, session factory, Repository[T], Alembic env
│   ├── llm/             Router, providers/, cost tracking, structured output
│   ├── eval/            Harness, metrics/, golden-set runner, drift
│   ├── obs/             structlog config, tracing decorator, counters
│   └── deploy/          Dockerfile / render.yaml / vercel.json templates
├── ui/                  Next.js component package (published separately)
└── tests/
```

## LLM router
Tier-based, not provider-based. Callers ask for "fast" or "smart" and never
name a provider — so swapping providers is a one-line config change.

```
"fast"  → Gemini Flash → Groq fallback → Claude Haiku fallback
"smart" → Claude → Gemini Pro fallback
```
Every call records tokens and cost through spine.obs.record_cost.

## Database
One Supabase project, one schema per app (masar, rasid, hisbah, dhawq,
bidaya) plus a shared `common` schema. Extensions: postgis (MASAR),
pgvector (HISBAH, DHAWQ, BIDAYA).

## UI package
Design tokens as CSS variables. Every component supports dark/light and
RTL from day one — retrofitting Arabic later costs far more than building
it in now.

## Deploy topology
```
Vercel (Next.js)  ──HTTPS──▶  Render (FastAPI)  ──▶  Supabase (Postgres)
                                     │
                                     └──▶  LLM providers
```

## Explicitly out of scope
No business logic. No domain models. No project-specific UI. If it serves
one app only, it lives in that app.
