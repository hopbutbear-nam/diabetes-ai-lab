# Nhật ký thí nghiệm

| Ngày | Mô hình | Config | AUC | Macro-F1 | Ghi chú |
|---|---|---|---|---|---|
| 2026-11-05 | majority | configs/train_majority.yaml | 0.500 (val) | 0.305 (val) | seed=42; val |
| 2026-11-05 | logreg | configs/train_logreg.yaml | 0.772 (val) | 0.424 (val) | seed=42; val |
| 2026-11-05 | xgb | configs/train_xgb.yaml | 0.781 (val) | 0.450 (val) | seed=42; val; best_iteration=168; backend=xgboost |
| 2026-11-05 | mlp | configs/train_mlp.yaml | 0.784 (val) | 0.405 (val) | seed=42; val; best_epoch=13 |
| 2026-11-06 | logreg | configs/evaluate.yaml | 0.772 | 0.457 | hiệu chuẩn, val; ECE 0.2567→0.0021; macro-F1 scaled=0.4566 |
| 2026-11-06 | xgb | configs/evaluate.yaml | 0.781 | 0.468 | hiệu chuẩn, val; ECE 0.1654→0.0016; macro-F1 scaled=0.4675 |
| 2026-11-06 | mlp | configs/evaluate.yaml | 0.784 | 0.470 | hiệu chuẩn, val; ECE 0.0046→0.0016; macro-F1 scaled=0.4699 |
| 2026-11-07 | majority | configs/evaluate.yaml | 0.500 | 0.305 | TEST (1 lần); multiclass_full; baseline |
| 2026-11-07 | logreg | configs/evaluate.yaml | 0.772 | 0.459 | TEST (1 lần); multiclass_full; CI AUC=[0.765, 0.778] |
| 2026-11-07 | xgb | configs/evaluate.yaml | 0.787 | 0.468 | TEST (1 lần); multiclass_full; CI AUC=[0.781, 0.793] |
| 2026-11-07 | mlp | configs/evaluate.yaml | 0.786 | 0.464 | TEST (1 lần); multiclass_full; mô hình tốt nhất; CI AUC=[0.780, 0.792] |
| 2026-11-07 | xgb | configs/evaluate.yaml | 0.824 | 0.687 | TEST (1 lần); binary_full; mô hình tốt nhất; CI AUC=[0.820, 0.829] |
| 2026-11-07 | mlp | configs/evaluate.yaml | 0.774 | 0.465 | TEST (1 lần); multiclass_dedup; mô hình tốt nhất; CI AUC=[0.768, 0.781] |
