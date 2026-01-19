VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup install test lint clean

help:
	@echo "Targets:"
	@echo "  make setup   - create venv + install (editable) + dev deps"
	@echo "  make install - install package (editable) into existing venv"
	@echo "  make test    - run tests"
	@echo "  make lint    - run ruff"
	@echo "  make clean   - remove venv and caches"

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -e ".[dev]"

install:
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache **/__pycache__
