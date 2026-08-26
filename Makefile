.PHONY: setup test lint all clean

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

all: lint test

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
