---
description: Close out the current session cleanly
---
Do these in order, stopping if any step fails:

1. Run `ruff check . --fix` and `ruff format .`
2. Run `pytest -q`. If anything fails, fix it before continuing.
3. Append an entry to docs/SESSION-LOG.md using this format:

   ## <YYYY-MM-DD> — <session title>
   **Scope:** <one line>
   **Built:** <bullets>
   **Decisions:** <any choice a future session needs to know about, and why>
   **Deferred:** <anything cut, and where it is tracked>
   **Next session should:** <one line>

4. Update README.md if the public behaviour or setup steps changed.
5. Print a one-paragraph summary of what changed. Do not commit.
