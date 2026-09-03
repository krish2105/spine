---
description: Verify this repo meets the definition of done
---
Check each item and report PASS or FAIL with evidence. Do not fix
anything — just report honestly.

1. `pytest -q` passes
2. `ruff check .` is clean
3. README.md has: problem, architecture diagram, stack, setup steps,
   results table, limitations, v2 backlog
4. docs/EVAL.md contains real measured numbers, not placeholders,
   and includes at least one baseline comparison
5. No secrets, API keys, or credentials in tracked files
6. Backend health endpoint responds locally
7. Frontend builds without errors
8. docs/SESSION-LOG.md covers every session so far

Finish with a blunt verdict: SHIP or NOT READY, and if not ready,
the shortest path to ready.
