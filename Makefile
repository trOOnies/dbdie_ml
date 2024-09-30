.PHONY: help venv install folders fmt lint clean-lint test clean-test clean-pyc build clean-build clean api rr
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

COLOUR_GREEN=\033[0;32m
COLOUR_RED=\033[0;31m
COLOUR_BLUE=\033[0;34m
END_COLOUR=\033[0m

help: ## Show the help
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

venv: ## Create venv
	@if [ -d ".venv" ]; then\
		echo "${COLOUR_RED}The venv already exists, please delete it manually first${END_COLOUR}";\
		false;\
	else\
		python3 -m venv .venv;\
	fi;
	@./.venv/bin/pip install -U pip
	@echo
	@echo "${COLOUR_GREEN}Use the command: source .venv/bin/activate${COLOUR_GREEN}"

install: ## Install the dependencies
	python3 -m pip install -r requirements.txt
	python3 -m pip install -r requirements_test.txt
	python3 -m pip install ../dbdie_classes

folders: ## Create the DBDIE folder structure
	python3 -m folder_structure

fmt: ## [ruff] Format the code
	ruff format

lint: ## [ruff] Run lint checks
	ruff check --output-format=concise

clean-lint: ## [ruff] Remove cache
	rm -rf .ruff_cache

test: ## [pytest] Test the code and coverage
	python3 -m pytest --cov=dbdie_ml

clean-test: ## [pytest] Remove test and coverage artifacts
	rm -rf .pytest_cache
	rm -rf .coverage

clean-pyc: ## Remove Python compiled bytecode files
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name 'pycache' -exec rm -fr {} +

build: ## Build the package
	python3 -m build

clean-build: ## Delete all build-related artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf dbdie_ml.egg-info/

clean: ## Remove all lint, test, coverage, compiled Python and build artifacts
	make clean-lint
	make clean-test
	make clean-pyc
	make clean-build

api: ## [fastapi] Run the API on localhost
	uvicorn --host=127.0.0.1 --port 8001 --app-dir=dbdie_ml --env-file=.env main:app

rr: ## Run the API after installing dependencies
	clear
	make install
	make api
