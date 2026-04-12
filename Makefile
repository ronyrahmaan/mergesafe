.PHONY: install test lint format clean run-experiment scan help

PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv sync --all-extras

test: ## Run tests
	$(PYTEST) tests/ -v --tb=short

test-cov: ## Run tests with coverage
	$(PYTEST) tests/ -v --cov=mergesafe --cov-report=term-missing

lint: ## Run linter
	$(RUFF) check src/ tests/

format: ## Format code
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

typecheck: ## Run type checker
	uv run mypy src/mergesafe/

clean: ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Experiment targets
run-experiment: ## Run single experiment (args: MODEL, ATTACK, METHOD)
	$(PYTHON) scripts/run_experiment.py \
		--model $(or $(MODEL),meta-llama/Llama-3.2-1B) \
		--attack $(or $(ATTACK),badnets) \
		--method $(or $(METHOD),ties)

run-all: ## Run full experiment matrix
	$(PYTHON) scripts/run_experiment.py --config configs/default.yaml --all-methods

scan: ## Scan adapters (args: ADAPTERS)
	$(PYTHON) -m mergesafe.cli scan $(ADAPTERS)

inject: ## Inject backdoor (args: MODEL, ATTACK)
	$(PYTHON) -m mergesafe.cli inject $(or $(MODEL),meta-llama/Llama-3.2-1B) \
		--attack $(or $(ATTACK),badnets)
