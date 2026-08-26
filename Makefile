.PHONY: setup test lint proof docs all clean

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

proof:
	uv run python scripts/proof.py

docs: proof
	uv run python docs/build.py

all: lint test docs

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ .coverage
