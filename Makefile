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

.PHONY: activate
activate:
	source .venv/bin/activate

.PHONY: fmt
fmt:
	black .

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: build
build:
	python -m build
