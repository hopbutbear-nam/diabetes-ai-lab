.PHONY: install data train evaluate serve test lint
install:
	pip install -r requirements-dev.txt && pip install -e .
data:
	python scripts/download_data.py && python scripts/prepare_data.py
train:
	python scripts/train.py --config configs/train_xgb.yaml
evaluate:
	python scripts/evaluate.py
serve:
	uvicorn diabetes_ai.serving.app:app --reload --port 8200
test:
	pytest
lint:
	ruff check src tests && ruff format --check src tests
