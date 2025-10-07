.PHONY: setup lint fmt test run_all

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
	git submodule update --init --recursive

lint:
	ruff check .

fmt:
	black .

test:
	pytest -q

run_all:
	bash scripts/run_all.sh
