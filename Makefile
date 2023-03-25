PY ?= .venv/bin/python

.PHONY: setup train demo figures serve test clean
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

train:  ## generate data, run SQL + backtest + forecast, write artifacts/model_card.md
	MLFLOW_DISABLE_AGENT_HINT=1 $(PY) -m pipelinetruth.pipeline

demo:  ## one command: regenerate data, train, print top deals, open the model card
	MLFLOW_DISABLE_AGENT_HINT=1 $(PY) -m pipelinetruth.pipeline --demo

figures:  ## refresh the README figures from the latest run
	mkdir -p docs && cp artifacts/reliability.png artifacts/forecast_backtest.png docs/

serve:  ## scoring API on http://127.0.0.1:8000 (docs at /docs)
	$(PY) -m uvicorn pipelinetruth.api:app --host 127.0.0.1 --port 8000

test:
	$(PY) -m pytest -q

clean:
	rm -rf data artifacts .pytest_cache
