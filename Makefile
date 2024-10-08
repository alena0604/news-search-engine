
# Variables
PYTHON = python
PRE_COMMIT = pre-commit
RUFF = ruff
BYTEWAX = bytewax.run
STREAMLIT = streamlit


# Targets
.PHONY: all format linter run_backend run_frontend help

# Default target
all: help

# Format code using pre-commit hooks
format:
	@$(PRE_COMMIT) run -a

# Lint code using ruff
linter:
	@$(RUFF) check --select I --fix
	@$(RUFF) format .

# Run the backend service
run_backend:
	@echo "Running backend..."
	@set -a && source .env && RUST_BACKTRACE=1 $(PYTHON) -m $(BYTEWAX) "./backend/flow:build"

# Run the frontend service
run_frontend:
	@echo "Starting frontend..."
	$(STREAMLIT) run ./frontend/ui.py

# Help target to display available commands
help:
	@echo "Makefile commands:"
	@echo "  make format        Format code using pre-commit hooks"
	@echo "  make linter        Lint code using ruff"
	@echo "  make run_backend   Run the backend service"
	@echo "  make run_frontend  Run the frontend service"
	@echo "  make help          Display this help message"
