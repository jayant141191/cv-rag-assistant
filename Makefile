.PHONY: install dev ingest run eval eval-fast test lint

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt && pip install -e .

ingest:
	python scripts/ingest.py data/sample_cvs

run:
	streamlit run app/streamlit_app.py

eval:
	python scripts/evaluate.py

eval-fast:
	python scripts/evaluate.py --no-judge

test:
	pytest -q

lint:
	ruff check .
