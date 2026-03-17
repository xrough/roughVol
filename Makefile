VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup install test lint clean proto-python serve

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

proto-python:
	mkdir -p generated/python
	$(PY) -m grpc_tools.protoc \
		-I proto \
		--python_out=generated/python \
		--grpc_python_out=generated/python \
		proto/rough_pricing.proto
	@echo "Stubs written to generated/python/"

serve:
	$(PY) -m roughvol.service.server
