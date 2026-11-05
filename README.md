# Diabetes AI Lab

Dự án học tập: sàng lọc **nguy cơ** đái tháo đường từ chỉ số sức khỏe (BRFSS 2015),
so sánh Logistic Regression / XGBoost / MLP, kèm giải thích, kiểm tra công bằng,
luật an toàn và API demo.

> ⚠️ Chỉ phục vụ học tập và nghiên cứu. Không chẩn đoán, không thay thế bác sĩ.

## Cấu trúc
Xem `docs/architecture.md`.

## Bắt đầu nhanh
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt && pip install -e .
python scripts/download_data.py        # đọc data/README.md trước
python scripts/prepare_data.py
python scripts/train.py --config configs/train_xgb.yaml
python scripts/evaluate.py
uvicorn diabetes_ai.serving.app:app --reload --port 8200
```

## Giao thức đánh giá (bắt buộc)
- Chia train / validation / test phân tầng; chọn ngưỡng trên validation, KHÔNG đụng test.
- Báo cáo ROC-AUC, PR-AUC, recall/precision từng lớp, macro-F1, calibration. Không dùng accuracy làm thước đo chính.
- Kiểm tra công bằng theo tuổi, giới, thu nhập.
