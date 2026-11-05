#!/usr/bin/env bash
set -euo pipefail
python scripts/prepare_data.py
for c in logreg xgb mlp; do python scripts/train.py --config configs/train_$c.yaml; done
python scripts/evaluate.py
