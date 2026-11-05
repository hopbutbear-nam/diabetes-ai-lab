

## 1. VAI TRÒ

Bạn là kỹ sư ML trưởng kiêm quản lý dự án kỹ thuật. Nhiệm vụ của bạn **không phải viết toàn bộ code**, mà viết **một bản kế hoạch triển khai cực kỳ chi tiết** (`docs/plan.md`) để một AI lập trình khác (Google Antigravity, sau đây gọi là "agent thực thi") đọc và làm từng khối một thay con người. Agent thực thi **không nhớ cuộc trò chuyện này**, mỗi khối nó nhận phải tự đủ thông tin để làm và tự kiểm chứng được.

## 2. BỐI CẢNH DỰ ÁN

Dự án học tập (sinh viên PTIT, môn Phát triển hệ thống thông minh): huấn luyện AI **sàng lọc nguy cơ đái tháo đường** từ 21 chỉ số sức khỏe khảo sát (BRFSS 2015). Đây **không** phải công cụ chẩn đoán y khoa; mọi kết quả hiển thị cho người dùng phải kèm miễn trừ trách nhiệm.

Môi trường người dùng:
- Windows, dùng **Git Bash**; đường dẫn dự án có **dấu tiếng Việt và khoảng trắng** (`F:\PT các HTTM\AITRAINING\diabetes-ai-lab`), nên mọi lệnh phải đặt đường dẫn trong dấu nháy và mã nguồn dùng `pathlib`.
- Đã có `.venv`, đã chạy `pip install -r requirements-dev.txt` và `pip install -e .`, `pytest` hiện đạt 2 test (`tests/test_schema.py`, `tests/test_api_health.py`).
- Python có thể là **3.14**: nhiều thư viện nặng (`torch`, `xgboost`, `shap`) có thể chưa có bản cài sẵn. Khối đầu tiên của kế hoạch phải kiểm tra việc này và nêu phương án dự phòng (Python 3.12 qua `py -3.12 -m venv .venv`, hoặc LightGBM/sklearn thay thế).
- GPU không đảm bảo có; mọi thứ phải chạy được trên CPU trong thời gian hợp lý (mỗi lần huấn luyện ≤ 10 phút, nêu cách giảm mẫu khi gỡ lỗi).
- Repo đã được dựng bằng `setup_repo.sh` (148 file, phần lớn là stub chỉ có docstring). Đã hoàn thành các bước: tạo cấu trúc, cài môi trường, tải CSV vào `data/raw/brfss2015/`. **Bước hiện tại là A.4.**

## 3. CÁC FILE ĐẦU VÀO VÀ CÁCH ĐỌC

Đọc kỹ cả ba file **trước khi viết bất cứ dòng kế hoạch nào**.

1. **`setup_repo.sh`**: nguồn sự thật về cấu trúc repo. Liệt kê chính xác mọi đường dẫn file/thư mục, nội dung các file đã có sẵn (`schema.py`, `configs/*.yaml`, `tests/*`, `Makefile`, `pyproject.toml`, `.gitignore`...). Kế hoạch **chỉ được dùng đường dẫn có trong script này**; nếu cần thêm file mới, ghi rõ "TẠO MỚI" và lý do. Không đổi tên file đã có.
2. **`diabetes_012_health_indicators_BRFSS2015.csv`**: dữ liệu thật. **Phải thực sự kiểm tra** (chạy pandas nếu có công cụ, hoặc đọc mẫu): số dòng, số cột, tên cột, kiểu dữ liệu, giá trị nhỏ nhất/lớn nhất từng cột, số ô thiếu, số dòng trùng lặp, phân bố nhãn `Diabetes_012`. Mọi con số trong kế hoạch về dữ liệu phải **lấy từ file thật, không suy đoán**. Nếu không chạy được code, ghi rõ "chưa kiểm chứng" và biến việc kiểm chứng thành bước đầu tiên của khối dữ liệu.
3. **`234.pdf`**: project của một bạn học (hệ đa agent cho bệnh tiểu đường). **Đây là tài liệu tham khảo, không phải đáp án để sao chép.** Cần rút ra:
   - Cấu hình huấn luyện: MLP 21→128→64→32→3, ReLU, dropout 0,15/0,1, Adam, batch 1.024, 14 epoch, cross-entropy có trọng số lớp (mũ 0,75), chọn epoch theo AUC validation, chia 80/20 phân tầng và lấy 30.442 hồ sơ từ tập train làm validation, hệ số điều chỉnh ngưỡng (1,0 / 1,5 / 0,9) chọn trên validation.
   - Kết quả tham chiếu trên tập test 50.736 hồ sơ: Logistic Regression AUC 0,773 / macro-F1 0,427; MLP AUC 0,782 / macro-F1 0,445–0,461; recall theo lớp 80% / 17% / 55%; precision nhóm tiền tiểu đường 4,0%; accuracy 75,2% **thấp hơn** baseline luôn đoán "không" (84,2%).
   - Cách giải thích: độ quan trọng hoán vị, tác động từng yếu tố (thay bằng trung bình dân số), what-if lối sống.
   - Khung đánh giá độ tin cậy: hiệu chuẩn xác suất (ECE), công bằng theo tuổi/giới/thu nhập với khoảng tin cậy Wilson 95%, chênh lệch recall theo nhóm tuổi 49,2 điểm %.
   - Luật an toàn 10 luật (R1–R10) và Bác sĩ phản biện (4 phép thử C1–C4).
   - Các hạn chế tác giả tự nêu. Kế hoạch của bạn phải **khắc phục hoặc kiểm tra lại** các điểm yếu này, không lặp lại chúng.

   Các con số của `234.pdf` chỉ dùng làm **mốc sanity check** (sai lệch cho phép ±0,02 AUC). Không đặt chúng làm mục tiêu cứng, và không được bịa kết quả; mọi con số kết quả thật phải do agent thực thi đo được trên máy.

## 4. PHẠM VI: TỪ BƯỚC A.4 ĐẾN HOÀN THÀNH

Danh sách công việc phải phủ hết (bạn được gộp và sắp xếp vào các khối ở mục 5, nhưng không được bỏ mục nào ở nhóm bắt buộc):

**Bắt buộc**
- A.0 Kiểm tra môi trường (phiên bản Python, thư viện nặng, pytest, đường dẫn có dấu cách).
- A.4 `data/load.py`, `data/validate.py` (đọc CSV, kiểm tra 21 cột + nhãn, thiếu, trùng, ngoài miền).
- A.5 `data/preprocess.py`, `data/split.py` (chia phân tầng, **xử lý dòng trùng lặp có chủ đích**, lưu chỉ số vào `data/splits/`, bài toán 3 lớp và biến thể nhị phân theo `configs/data.yaml`).
- A.6 Test chống rò rỉ dữ liệu (`tests/test_split_no_leakage.py`, `test_preprocess.py`).
- B Phân tích khám phá (`notebooks/01_eda_brfss.ipynb`, hình lưu vào `reports/figures/`).
- C.1 `features/` (chuẩn hóa fit trên train, lưu mean/std), `evaluation/metrics.py`.
- C.2 Baseline (majority, Logistic Regression) + ghi `docs/experiment_log.md`.
- C.3 XGBoost.
- C.4 MLP PyTorch (`models/mlp.py`, `training/trainer.py`, `losses.py`, `callbacks.py`, `training/train.py`, `scripts/train.py`).
- D.1 Chọn ngưỡng trên validation (`evaluation/thresholds.py`).
- D.2 Hiệu chuẩn (`calibration.py`), khoảng tin cậy (`confidence_intervals.py`).
- D.3 Công bằng theo nhóm (`fairness.py`), đánh giá test **một lần duy nhất** (`scripts/evaluate.py`, `evaluation/report.py`, `plots.py`).
- D.4 Giải thích: `explain/permutation.py`, `shap_explain.py`, `what_if.py`.
- E.1 Luật an toàn (`safety/rules.py`, `risk_level.py`, `disclaimers.py`, đọc `configs/safety_rules.yaml`).
- E.2 `inference/` (`schemas.py`, `predictor.py`, `export.py`, `scripts/export_model.py`).
- E.3 API `serving/` (`/health`, `/predict`, `/explain`) + test.
- E.4 Demo web `app/` (form nhập 21 chỉ số, hiển thị xác suất, yếu tố, cảnh báo, miễn trừ trách nhiệm).
- E.5 Tài liệu: `reports/model_card.md`, `fairness_report.md`, `final_report.md`, `docs/architecture.md`, `data_dictionary.md`, `safety_and_ethics.md`, README cập nhật, Makefile/`run_all.sh` chạy được từ đầu đến cuối.

**Tùy chọn (đánh dấu rõ "TÙY CHỌN", chỉ làm khi nhóm bắt buộc đã xong và đạt)**
- `safety/critic.py` (Bác sĩ phản biện C1–C4), `training/tuning.py`, `cross_validation.py`, `knowledge/`, `agents/`.

## 5. QUY TẮC CHIA KHỐI (QUAN TRỌNG NHẤT)

Agent thực thi bị giới hạn mức sử dụng mỗi phiên, nhưng người dùng muốn **ít khối, mỗi khối lớn hơn**. Vì vậy:

1. **Đúng khoảng 10 khối bắt buộc (K01–K10)**, cộng tối đa **1 khối tùy chọn (K11)** cho phần TÙY CHỌN. Không tạo thêm khối khác. Mục A.0 (kiểm tra môi trường) được gộp vào K01.
2. **Mỗi khối tối đa 15 file tạo/sửa** (tính cả file test, file cấu hình, notebook, tài liệu). Khối lượng các khối phải **gần bằng nhau**: mỗi khối nên có từ 8 đến 15 file, tổng **≤ ~800 dòng code mới** (không tính dòng trống/docstring), **≤ 8 lệnh kiểm chứng**. Không khối nào được vượt 15 file; nếu một nhóm việc vượt, chuyển bớt sang khối liền kề thay vì tạo thêm khối.
3. **Gợi ý phân bổ** (bạn được điều chỉnh để cân bằng khối lượng, nhưng phải tôn trọng thứ tự phụ thuộc):
   - K01: A.0 + A.4 + A.5 + A.6 (môi trường, load, validate, preprocess, split, test chống rò rỉ).
   - K02: B + C.1 (EDA, features/scalers, metrics, test).
   - K03: C.2 + C.3 (baseline, XGBoost, registry, train script, ghi experiment_log).
   - K04: C.4 (MLP, trainer, losses, callbacks, train script, test).
   - K05: D.1 + D.2 (ngưỡng, hiệu chuẩn, khoảng tin cậy).
   - K06: D.3 (công bằng, đánh giá test một lần, plots, report, `scripts/evaluate.py`).
   - K07: D.4 (permutation, SHAP, what-if, test).
   - K08: E.1 + E.2 (luật an toàn, risk_level, predictor, export).
   - K09: E.3 + E.4 (API, test API, demo web).
   - K10: E.5 (model card, báo cáo, tài liệu, README, `run_all.sh`, chạy toàn pipeline từ đầu đến cuối).
   - K11 (TÙY CHỌN): critic, tuning, cross-validation, knowledge, agents.
4. **Vì khối lớn, mỗi khối phải được chia thành 3–5 "bước con" (a, b, c...)** theo thứ tự thực hiện. Mỗi bước con có: danh sách file, kiểm chứng nhỏ (lệnh + kết quả mong đợi) và một **commit trung gian**. Mục đích: nếu agent hết lượt giữa chừng, lần sau chỉ cần tiếp tục từ bước con chưa xong. Mỗi bước con không quá 5 file.
5. Mỗi khối ghi nhãn kích cỡ theo số file: **S** (1–5), **M** (6–10), **L** (11–15). Cố gắng để ≥ 80% khối là cỡ L hoặc M cao.
6. Mỗi khối **tự đủ**: nêu rõ phải đọc những file nào trước khi làm, và khối trước đó phải để lại những gì (artifact, hàm, file). Không viết "như đã nói ở trên".
7. Mỗi khối kết thúc bằng một **commit tổng kết** theo chuẩn Conventional Commits (ngoài các commit trung gian của bước con).
8. Thứ tự phải tôn trọng phụ thuộc kỹ thuật: chia dữ liệu trước mọi việc; chuẩn hóa chỉ fit trên train; baseline trước mô hình phức tạp; ngưỡng chỉ chọn trên validation; test chỉ chạm một lần ở bước đánh giá cuối.

## 6. MẪU BẮT BUỘC CHO MỖI KHỐI

Mỗi khối trong `plan.md` phải theo đúng mẫu sau (không bỏ mục nào):

```
### K<số> — <Tên khối>  [Cỡ: S/M/L, <số file> file]  [Bắt buộc / Tùy chọn]
**Mục tiêu:** 1–3 câu.
**Phụ thuộc:** các khối phải xong trước (K..), artifact cần có.
**Đọc trước khi làm:** danh sách file (đường dẫn đầy đủ).
**Tổng hợp file của khối:** bảng `STT | Đường dẫn | TẠO MỚI/SỬA | Bước con | Vai trò`
  (tối đa 15 dòng).

#### Bước con K<số>a — <tên>
**File:** ... (tối đa 5)
**Đặc tả code chi tiết** (cho từng file):
  - Tên hàm/lớp, chữ ký đầy đủ với type hint, tham số mặc định.
  - Logic từng bước (đủ cụ thể để viết ngay, ví dụ: "dùng
    sklearn.model_selection.train_test_split(stratify=y, random_state=seed)").
  - Đầu vào / đầu ra, định dạng file lưu, quy ước đặt tên.
  - Lỗi cần bắt và thông báo lỗi (tiếng Việt, nêu cột/giá trị gây lỗi).
  - Đoạn code mẫu ngắn cho phần dễ sai (không dán cả file).
**Cấu hình liên quan:** khóa trong `configs/*.yaml` cần đọc/sửa.
**Kiểm chứng bước con:** lệnh chính xác (Git Bash, đường dẫn có dấu nháy) +
  kết quả mong đợi cụ thể.
**Commit trung gian:** `<type>(<scope>): <mô tả>`.

(lặp lại cho K<số>b, K<số>c, ...)

#### Kiểm thử và tiêu chí hoàn thành của cả khối
**Test tự động cần viết:** tên file, tên hàm test, assert cụ thể.
**Tiêu chí hoàn thành (định lượng, kiểm chứng được):**
  - Ví dụ: "pytest: 14 passed", "train+val+test = 253.680 dòng, giao nhau = 0",
    "tỉ lệ lớp 2 ở mỗi tập nằm trong 13,9% ± 0,3 điểm %",
    "AUC OvR validation của LogReg trong [0,75; 0,80]".
**Artifact tạo ra:** file/đường dẫn để khối sau dùng.
**Ghi nhật ký:** dòng cần thêm vào `docs/experiment_log.md` (nếu có số liệu).
**Commit tổng kết:** `<type>(<scope>): <mô tả>`.
**Rủi ro & cách xử lý:** ≥ 3 mục (ví dụ lỗi encoding, thiếu RAM, torch không cài được).
**Lệnh giao cho agent thực thi:** 5–10 dòng tiếng Việt, tự đứng được, có thể dán
  nguyên vào Antigravity; nêu rõ "làm lần lượt các bước con a, b, c...; sau mỗi
  bước con chạy kiểm chứng và commit; nếu bước con nào thất bại thì dừng và báo lỗi,
  không tự sửa dữ liệu hay tiêu chí".
```

## 7. YÊU CẦU NỘI DUNG KỸ THUẬT (BẮT BUỘC TUÂN THỦ TRONG KẾ HOẠCH)

**Dữ liệu và chia tập**
- Nêu quy tắc chia cụ thể, nhất quán với `configs/data.yaml` và `234.pdf`: test 20%; validation lấy 15% từ phần còn lại (hoặc nêu chính xác công thức và số dòng kết quả); phân tầng theo nhãn; `seed=42`. Tính ra số dòng từng tập từ dữ liệu thật.
- **Dòng trùng lặp**: đo số lượng thật trong CSV, rồi quyết định có xóa/nhóm trước khi chia hay không, giải thích rủi ro rò rỉ giữa train và test, và ghi quyết định vào `docs/data_dictionary.md`. Nếu chọn giữ trùng lặp để so sánh được với `234.pdf`, phải nêu rõ và thêm một thí nghiệm đối chứng đã khử trùng.
- Lưu chỉ số chia (không lưu bản sao dữ liệu) vào `data/splits/`; có test chứng minh ba tập không giao nhau và hợp lại đủ tổng số dòng.
- Hỗ trợ cả bài toán 3 lớp và nhị phân (tiền tiểu đường + tiểu đường = 1) qua `configs/data.yaml`, và kế hoạch nêu rõ cách chạy từng phiên bản.

**Huấn luyện**
- Mọi siêu tham số đọc từ `configs/*.yaml`, không viết cứng trong code. Seed cố định qua `seed.py`, đảm bảo tái lập (chạy hai lần cho cùng kết quả trong sai số nêu rõ).
- Xử lý mất cân bằng bằng trọng số lớp (mũ 0,75 như `234.pdf`), chỉ áp dụng trên train. Nêu cách tính trọng số.
- Chọn epoch/ngưỡng chỉ bằng validation; ghi nhận epoch tốt nhất.
- Tách rõ: baseline phải được đo trước, và **mỗi mô hình mới phải so với baseline trên cùng giao thức**.

**Đánh giá (nghiêm ngặt)**
- Thước đo chính: ROC-AUC one-vs-rest, PR-AUC, macro-F1, recall/precision từng lớp, Brier, ECE. **Không dùng accuracy làm thước đo chính**; luôn in kèm accuracy của baseline "luôn đoán lớp đa số" để người đọc thấy bối cảnh.
- Khoảng tin cậy bootstrap (n=1000) cho AUC và recall; khoảng Wilson 95% cho tỉ lệ theo nhóm.
- Công bằng: recall, precision, tỉ lệ cảnh báo theo nhóm tuổi (13 nhóm gộp hợp lý), giới, thu nhập; nêu cách gộp nhóm để mẫu đủ lớn; báo chênh lệch tối đa giữa các nhóm.
- Test chỉ chạm **một lần** trong `scripts/evaluate.py`; kế hoạch phải có bước kiểm tra tự động hoặc quy ước ngăn việc chỉnh tham số dựa trên test.
- Tiêu chí "đạt" của từng mô hình dựa trên khoảng sanity từ `234.pdf` nhưng có dung sai và cách xử lý khi lệch (ví dụ AUC thấp bất thường → kiểm tra rò rỉ, lỗi chuẩn hóa, thứ tự nhãn).

**Giải thích và an toàn**
- What-if chỉ trên các yếu tố có thể thay đổi (BMI, vận động, hút thuốc, rượu, rau, trái cây), loại các cột trong `drop_for_what_if` của `configs/features.yaml`. Ghi rõ đây là tương quan, không phải nhân quả.
- Luật an toàn đọc từ `configs/safety_rules.yaml`; luật mức "danger" luôn nâng mức nguy cơ. Nêu cách xử lý khi hồ sơ không có đường huyết (mô hình không nhận glucose làm đầu vào): luật chỉ chạy khi có dữ liệu tương ứng.
- Mức nguy cơ: chỉ số = P(tiểu đường) + 0,5·P(tiền tiểu đường); dưới 0,10 thấp, dưới 0,25 trung bình, còn lại cao (theo `configs/serve.yaml`). Mọi phản hồi API/web phải kèm miễn trừ trách nhiệm và khuyên xét nghiệm xác nhận.

**API và demo**
- Schema đầu vào có kiểm tra miền giá trị từng đặc trưng (Pydantic), trả lỗi 422 rõ ràng.
- Khi chưa có model, `/predict` trả lỗi có thông báo thân thiện thay vì sập.
- Demo web tĩnh gọi API cục bộ cổng 8200; xử lý CORS.

## 8. ĐỊNH DẠNG ĐẦU RA

Xuất **duy nhất một file**: `docs/plan.md`, tiếng Việt, Markdown, có cấu trúc:

1. **Tóm tắt dự án và nguyên tắc chung** (≤ 25 dòng): mục tiêu, môi trường, quy ước (seed, đường dẫn, lệnh, commit), cách một khối và một bước con được coi là "xong".
2. **Kết quả kiểm tra đầu vào**: bảng số liệu thật từ CSV (số dòng, cột, thiếu, trùng, phân bố nhãn, kích thước các tập sẽ chia), và nhận xét 5–10 dòng về những điểm cần thận trọng rút ra từ `234.pdf`.
3. **Các giả định** (đánh số), nếu có thông tin thiếu. **Không hỏi lại người dùng**; tự chọn phương án hợp lý nhất và ghi giả định.
4. **Bảng tổng quan các khối** (đúng 10 khối bắt buộc + tối đa 1 tùy chọn): cột `ID | Tên | Cỡ | Số file | Bắt buộc/Tùy chọn | Phụ thuộc | Tiêu chí hoàn thành ngắn`.
5. **Sơ đồ phụ thuộc** dạng danh sách hoặc Mermaid.
6. **Chi tiết từng khối** K01…K10 (và K11 nếu có) theo đúng mẫu mục 6.
7. **Bảng theo dõi tiến độ** (checklist `[ ]` cho từng khối **và từng bước con**, kèm cột ghi kết quả đo được).
8. **Phụ lục**: (a) bảng ánh xạ file → khối phụ trách, đảm bảo **mọi file stub quan trọng trong `setup_repo.sh` được gán cho một khối hoặc ghi rõ "bỏ qua + lý do"**; (b) danh sách lệnh thường dùng; (c) bảng lỗi thường gặp trên Windows/Git Bash và cách sửa; (d) bảng so sánh "kết quả tham chiếu từ 234.pdf" với "kết quả cần đo", để trống cột đo được.

## 9. TIÊU CHÍ CHẤT LƯỢNG CỦA CHÍNH KẾ HOẠCH (TỰ KIỂM TRƯỚC KHI NỘP)

Trước khi xuất file, tự kiểm tra và sửa:
- [ ] Có đúng 10 khối bắt buộc (K01–K10), tối đa 1 khối tùy chọn K11; không có khối nào khác.
- [ ] Không khối nào vượt 15 file; mỗi khối có 3–5 bước con, mỗi bước con ≤ 5 file; khối lượng các khối tương đối đồng đều.
- [ ] Mọi đường dẫn file đều tồn tại trong `setup_repo.sh` hoặc được đánh dấu TẠO MỚI.
- [ ] Mọi khối và bước con có đủ các mục của mẫu.
- [ ] Mọi tiêu chí hoàn thành đều **đo được bằng lệnh** (không dùng từ mơ hồ như "chạy tốt", "hợp lý").
- [ ] Mọi con số về dữ liệu đều lấy từ CSV thật; mọi con số từ `234.pdf` được gắn nhãn "tham chiếu".
- [ ] Không có bước nào dùng dữ liệu test để chọn mô hình, ngưỡng hoặc siêu tham số.
- [ ] Không có khối hay bước con nào phụ thuộc vào thứ đứng sau nó.
- [ ] Mọi lệnh chạy được trong Git Bash trên Windows với đường dẫn có khoảng trắng.
- [ ] Kế hoạch có mục xử lý lỗi cho: `torch` không cài được trên Python 3.14, thiếu RAM, encoding, `pytest` không thấy package (`pythonpath = ["src"]`).
- [ ] Phần tùy chọn không chặn đường găng của phần bắt buộc.

## 10. RÀNG BUỘC CUỐI

- Không bịa số liệu, không giả vờ đã chạy code nếu chưa chạy. Điều gì chưa kiểm chứng thì ghi "chưa kiểm chứng" và biến nó thành bước kiểm chứng đầu tiên.
- Không viết toàn bộ mã nguồn trong kế hoạch; chỉ đặc tả đủ chi tiết và đoạn mẫu ngắn cho phần dễ sai.
- Không thêm công nghệ ngoài những gì đã có trong `requirements.txt` trừ khi có lý do và ghi rõ (ví dụ `optuna`, `lightgbm`).
- Hệ thống chỉ phục vụ học tập, không chẩn đoán; điều này phải xuất hiện trong kế hoạch dưới dạng bước kiểm tra cụ thể ở khối API, demo và báo cáo.
- Trả lời bằng tiếng Việt. Chỉ xuất nội dung file `docs/plan.md`, không thêm lời dẫn ngoài file.
