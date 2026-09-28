.PHONY: install test lint format run figures clean help
.DEFAULT_GOAL := help

PYTHON ?= python
OUT    ?= results

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev and matrix-profile extras
	$(PYTHON) -m pip install -e ".[dev,mp]"

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Check formatting and lint rules
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:  ## Apply formatting and autofixable lint rules
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

run:  ## Regenerate every CSV, figure and table into $(OUT)/
	$(PYTHON) -m tsadharness.cli run --out $(OUT)

figures: run  ## Alias for `run`; the figures are produced by the same command

clean:  ## Remove build artifacts and caches (leaves results/ alone)
	rm -rf build dist .pytest_cache .ruff_cache **/__pycache__
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
