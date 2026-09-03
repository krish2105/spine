# Module Contracts

The interfaces the five apps depend on. Changing anything here breaks
downstream repos — update this file in the same change, and note it in
SESSION-LOG.md.

## spine.auth
```python
get_current_user(request) -> User | None
require_role(role: str) -> Depends           # FastAPI dependency
class User: id: UUID; email: str; role: str
```

## spine.db
```python
class Base(DeclarativeBase): ...
get_session() -> AsyncSession                # FastAPI dependency
class Repository[T]:
    async def get(id) -> T | None
    async def list(**filters) -> list[T]
    async def create(obj: T) -> T
    async def update(id, **fields) -> T
    async def delete(id) -> None
```

## spine.llm
```python
class LLMRouter:
    async def complete(prompt, *, model_tier="fast", schema=None) -> LLMResponse
    async def stream(prompt, *, model_tier="fast") -> AsyncIterator[str]

# model_tier: "fast" | "smart"  — router picks the provider
class LLMResponse: text: str; parsed: Any | None; tokens_in: int
                   tokens_out: int; cost_usd: float; model: str
```

## spine.eval
```python
class Metric(Protocol):
    name: str
    def compute(self, predictions, targets) -> float

class EvalHarness:
    def register(self, metric: Metric) -> None
    async def run(self, golden_set, predict_fn) -> EvalReport

class EvalReport: metrics: dict[str, float]; failures: list[Case]
                  baseline_delta: dict[str, float]
```

## spine.obs
```python
get_logger(name) -> structlog.BoundLogger
@traced                                      # decorator, records latency
record_cost(tokens_in, tokens_out, model) -> None
```

## Design rules
- Everything async. No sync DB or LLM calls.
- Errors raise typed exceptions from spine.errors, never bare Exception.
- No module reads env vars directly — all config through spine.config.Settings.
