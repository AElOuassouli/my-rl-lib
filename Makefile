


run-mlflow: ## Opens MLflow UI
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

test: ## Run unit tests
	uv run pytest

test-cov: ## Run unit tests with coverage report
	uv run pytest --cov=my_rl_lib --cov-report=term-missing

format: ## Format code with ruff
	uv run ruff format .

type-check: ## Run mypy type checker
	uv run mypy src/

check: ## Run all checks (lint, format, type) — read-only
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/

fix: ## Auto-fix lint violations and reformat code
	uv run ruff check --fix .
	uv run ruff format .

install-hooks: ## Install git hooks (run once after cloning)
	git config core.hooksPath .githooks
	chmod +x .githooks/pre-commit .githooks/pre-push