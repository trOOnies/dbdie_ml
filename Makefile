COLOUR_GREEN=\033[0;32m
COLOUR_RED=\033[0;31m
COLOUR_BLUE=\033[0;34m
END_COLOUR=\033[0m

.PHONY: venv
venv:
	@if [ -d ".venv" ]; then\
		echo "${COLOUR_RED}The venv already exists, please delete it manually first${END_COLOUR}";\
		false;\
	else\
		python3 -m venv .venv;\
	fi;
	@./.venv/bin/pip install -U pip
	@echo
	@echo "${COLOUR_GREEN}Use the command: source .venv/bin/activate${COLOUR_GREEN}"

.PHONY: install
install:
	python3 -m pip install -r requirements.txt

.PHONY: folders
folders:
	python3 -m folder_structure

.PHONY: fmt
fmt:
	ruff format

.PHONY: lint
lint:
	ruff check --output-format=concise

.PHONY: test
test:
	python3 -m pytest --cov=dbdie_ml

.PHONY: build
build:
	python3 -m build
