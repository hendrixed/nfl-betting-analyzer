.PHONY: verify prepare coverage test lint type-check format clean

# Main verification target
# Prepares a minimal snapshot and artifacts, then runs tests, lint, and mypy.
verify: prepare test lint type-check

# Prepare minimal artifacts for CI and local verification
prepare:
	@echo "Preparing snapshot and artifacts..."
	python nfl_cli.py odds-snapshot --provider mock --max-offers 50
	python nfl_cli.py snapshot-verify --repair
	python nfl_cli.py backtest --target receiving_yards --market player_receiving_yards --limit 50
	$(MAKE) coverage

# Generate coverage matrices
coverage:
	@echo "Generating coverage matrices..."
	python generate_coverage_matrices.py

# Run tests
test:
	@echo "Running tests..."
	@mkdir -p reports
	python -m pytest -q --junitxml reports/junit.xml

# Run linting
lint:
	@echo "Running ruff linter..."
	ruff check .

# Run type checking
type-check:
	@echo "Running mypy type checking..."
	mypy --strict core api

# Format code
format:
	@echo "Formatting code..."
	ruff format .

# Clean cache files
clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Install development dependencies
dev-install:
	@echo "Installing development dependencies..."
	pip install ruff mypy pytest pytest-asyncio black
