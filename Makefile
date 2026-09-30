.PHONY: setup smoke test reproduce

setup:
	bash scripts/setup.sh

smoke:
	.venv/bin/saverouter smoke-test

test:
	.venv/bin/python -m pytest

reproduce:
	bash scripts/reproduce.sh all
