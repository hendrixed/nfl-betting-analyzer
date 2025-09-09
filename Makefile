.PHONY: verify test lint type-check format clean

# Main verification target
verify: test lint type-check

# Run tests
test:
	@echo "Running tests..."
	python -m pytest -q

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
