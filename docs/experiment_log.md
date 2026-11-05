# Nhật ký thí nghiệm

| Ngày | Mô hình | Config | AUC | Macro-F1 | Ghi chú |
|---|---|---|---|---|---|
| 2026-11-05 | majority | configs/train_majority.yaml | 0.500 (val) | 0.305 (val) | seed=42; val |
| 2026-11-05 | logreg | configs/train_logreg.yaml | 0.772 (val) | 0.424 (val) | seed=42; val |
| 2026-11-05 | xgb | configs/train_xgb.yaml | 0.781 (val) | 0.450 (val) | seed=42; val; best_iteration=168; backend=xgboost |
| 2026-11-05 | mlp | configs/train_mlp.yaml | 0.784 (val) | 0.405 (val) | seed=42; val; best_epoch=13 |
