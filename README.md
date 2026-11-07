# Diabetes AI Lab

Dự án học tập: Sàng lọc **nguy cơ** đái tháo đường từ chỉ số sức khỏe cộng đồng (BRFSS 2015), so sánh hiệu năng giữa Hồi quy Logistic, XGBoost và PyTorch MLP, tích hợp giải thích toàn cục, kiểm định công bằng nhóm, hệ thống luật an toàn lâm sàng và dịch vụ API/Web demo.

> ⚠️ **Miễn trừ trách nhiệm:** Hệ thống này được xây dựng **chỉ để học tập, nghiên cứu và thử nghiệm công nghệ; không phải chẩn đoán y khoa**. Mô hình không thể và không được dùng để thay thế ý kiến chuyên môn của bác sĩ hoặc các xét nghiệm y tế xác nhận chính thức.

---

## 1. Bắt đầu nhanh

Chạy toàn bộ pipeline một lệnh duy nhất bằng bash (hoặc Git Bash trên Windows):

```bash
# Khởi tạo môi trường ảo và cài đặt thư viện
python -m venv .venv
source .venv/bin/activate       # Trên Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements-dev.txt && pip install -e .

# Chạy toàn bộ quy trình từ tiền xử lý, huấn luyện đến đánh giá và xuất báo cáo
bash scripts/run_all.sh
```

Sau khi hoàn thành, khởi chạy dịch vụ demo và mở giao diện người dùng:
```bash
uvicorn diabetes_ai.serving.app:app --reload --port 8200
```
Truy cập trình duyệt tại:
- **Giao diện Web Demo:** [http://localhost:8200/app/](http://localhost:8200/app/)
- **Tài liệu OpenAPI / Swagger:** [http://localhost:8200/docs](http://localhost:8200/docs)

---

## 2. Hướng dẫn thực thi từng bước độc lập

Nếu muốn kiểm soát hoặc tinh chỉnh từng bước trong chu trình:

1. **Chuẩn bị dữ liệu và phân chia tập train/val/test:**
   ```bash
   python scripts/prepare_data.py
   ```
   *(Tạo phân chia stratified 60/20/20 cho cả bộ dữ liệu đầy đủ và bộ dữ liệu đã khử trùng lặp)*

2. **Huấn luyện mô hình:**
   ```bash
   # Huấn luyện mô hình Majority baseline
   python scripts/train.py --config configs/train_majority.yaml
   # Huấn luyện mô hình Logistic Regression
   python scripts/train.py --config configs/train_logreg.yaml
   # Huấn luyện mô hình XGBoost
   python scripts/train.py --config configs/train_xgb.yaml
   # Huấn luyện mô hình PyTorch MLP
   python scripts/train.py --config configs/train_mlp.yaml
   ```

3. **Hiệu chuẩn xác suất và tìm ngưỡng phân lớp tối ưu:**
   ```bash
   python scripts/calibrate_and_threshold.py
   ```
   *(Thực hiện hiệu chuẩn và tối ưu hóa Macro-F1 độc quyền trên tập validation)*

4. **Giải thích mô hình toàn cục:**
   ```bash
   python scripts/explain_global.py
   ```
   *(Tính toán Permutation Importance trên tập validation và xuất biểu đồ top đặc trưng)*

5. **Đánh giá kiểm thử (Tuân thủ giao thức khóa test):**
   ```bash
   # Đánh giá trên tập validation (chạy thoải mái để kiểm tra)
   python scripts/evaluate.py --split val

   # Đánh giá trên tập TEST (chỉ được chạy đúng 1 lần duy nhất trong toàn dự án)
   python scripts/evaluate.py
   ```

6. **Đóng gói mô hình phục vụ suy luận:**
   ```bash
   python scripts/export_model.py
   ```

7. **Tự động sinh các báo cáo chuẩn:**
   ```bash
   python scripts/make_reports.py
   ```

---

## 3. Chạy các biến thể thực nghiệm (Nhị phân & Khử trùng lặp)

Dự án cung cấp sẵn cấu hình và cờ hỗ trợ hai bài toán đối chứng:

### 3.1. Bài toán phân loại nhị phân (Binary Classification)
Gộp nhóm 0 (Không mắc) và nhóm 1 (Tiền tiểu đường) thành lớp âm tính (0), đối chiếu với nhóm 2 (Đái tháo đường - lớp 1):
```bash
# Huấn luyện các mô hình cho bài toán nhị phân
python scripts/train.py --config configs/train_xgb.yaml --binary
python scripts/train.py --config configs/train_mlp.yaml --binary

# Hiệu chuẩn và tối ưu ngưỡng cho bài toán nhị phân
python scripts/calibrate_and_threshold.py --binary
```

### 3.2. Tập dữ liệu khử trùng lặp (Deduplicated Dataset)
Loại bỏ 24.206 bản ghi có vector đặc trưng trùng lặp hoàn toàn khỏi dữ liệu gốc:
```bash
# Huấn luyện trên tập đã khử trùng lặp
python scripts/train.py --config configs/train_mlp.yaml --dedup
python scripts/train.py --config configs/train_xgb.yaml --dedup

# Hiệu chuẩn trên tập đã khử trùng lặp
python scripts/calibrate_and_threshold.py --dedup
```

---

## 4. Cấu trúc thư mục dự án

```text
diabetes-ai-lab/
├── configs/            # Tệp cấu hình YAML (dữ liệu, huấn luyện, đánh giá, an toàn, phục vụ)
├── data/               # Dữ liệu gốc (raw) và chỉ mục phân chia (splits/train/val/test)
├── docs/               # Tài liệu chi tiết: kiến trúc, an toàn đạo đức, tham khảo, kế hoạch
├── models/             # Trọng số mô hình đã huấn luyện (.joblib, .pt, .json) và scaler
├── reports/            # Báo cáo kết quả: model_card.md, fairness_report.md, final_report.md, figures/
├── scripts/            # Kịch bản dòng lệnh thực thi từng giai đoạn của pipeline
├── src/diabetes_ai/    # Mã nguồn thư viện chính:
│   ├── data/           # Nạp, kiểm tra toàn vẹn và phân chia dữ liệu
│   ├── features/       # Định nghĩa và chuẩn hóa đặc trưng (StandardScaler)
│   ├── models/         # Triển khai Majority, LogReg, XGBoost, PyTorch MLP
│   ├── evaluation/     # Metrics, hiệu chuẩn (Isotonic/Temperature), công bằng, khóa test
│   ├── explain/        # Giải thích độ quan trọng hoán vị (Permutation Importance)
│   ├── safety/         # Bộ luật an toàn lâm sàng (Red Flags, Extreme BMI, ghi đè rủi ro)
│   ├── inference/      # Lớp Predictor tích hợp tiền xử lý, mô hình và an toàn
│   └── serving/        # Ứng dụng web FastAPI và giao diện người dùng Web UI
└── tests/              # Bộ kiểm thử tự động (pytest)
```

---

## 5. Giao thức đánh giá (Bắt buộc)

- **Chia tách dữ liệu nghiêm ngặt:** Phân chia phân tầng Train (60%), Validation (20%), Test (20%). Toàn bộ quá trình chọn mô hình, điều chỉnh siêu tham số, hiệu chuẩn xác suất và tìm ngưỡng quyết định chỉ được thực hiện trên tập **Validation**. Tuyệt đối không đụng đến tập **Test**.
- **Cơ chế khóa kiểm thử một lần (`test_lock.json`):** Tập test chỉ được nạp và đánh giá đúng **MỘT LẦN DUY NHẤT** bằng `scripts/evaluate.py`. Hệ thống tự động băm mã SHA-256 các tệp mô hình và từ chối các lần chạy lại để ngăn chặn tối ưu hóa quá mức (*p-hacking*).
- **Thước đo đánh giá khách quan:**
  - Accuracy bị chi phối mạnh bởi mất cân bằng lớp (baseline "luôn đoán không" đạt ~84.7% accuracy), do đó **không dùng accuracy làm thước đo chất lượng chính**.
  - Các thước đo chủ đạo bao gồm: **ROC-AUC đa lớp (OVR/macro)**, **PR-AUC**, **Macro-F1**, **Recall/Precision từng lớp**, và độ sai lệch hiệu chuẩn **Expected Calibration Error (ECE)**.
  - Mọi chỉ số đều được báo cáo kèm khoảng tin cậy 95% tính toán bằng Bootstrap hoặc phương pháp Wilson.
- **Kiểm định công bằng phân tầng:** Đánh giá độc lập hiệu năng mô hình trên các nhóm nhân khẩu học (Độ tuổi, Giới tính sinh học, Mức thu nhập) để phát hiện và công khai minh bạch các chênh lệch tỷ lệ phát hiện.
