# Bộ prompt giao việc cho Antigravity — Diabetes AI Lab

Dựa trên `docs/plan.md` (K01–K11) và cấu trúc `diabetes-ai-lab.zip`. Gồm 35 prompt, mỗi prompt đúng một bước con; làm theo thứ tự.

## 1. Cách dùng

- **Một prompt = một cuộc trò chuyện mới** trong Antigravity. Xong thì đọc báo cáo, chạy `git log -1`, rồi mới mở cuộc mới cho prompt kế tiếp. Không gộp nhiều prompt vào một cuộc.
- `plan.md` nặng 177 KB. Mỗi prompt chỉ cho agent đọc đúng đoạn của bước đó (`sed -n 'A,Bp'`), nên ngữ cảnh luôn nhỏ. Trạng thái giữa các cuộc nằm ở **git** và **bảng tiến độ §7** của plan, không nằm trong trí nhớ của agent.
- Đi đúng thứ tự K01 → K10 (K11 tùy chọn). Khối sau dùng artifact của khối trước nên không nhảy cóc.
- Ba bước có lệnh chạy rất lâu (huấn luyện MLP ở P14, huấn luyện biến thể ở P20, chạy lại toàn pipeline ở P32): bạn có thể tự chạy lệnh trong terminal rồi dán khoảng 15 dòng cuối log cho agent kiểm tra. Agent không phải canh nên đỡ tốn token.
- Hết token giữa chừng: dùng prompt **TIẾP TỤC** ở cuối tài liệu.
- Số test tích lũy cuối mỗi khối: K01 = 18, K02 = 33, K03 = 40, K04 = 48, K05 = 60, K06 = 69, K07 = 77, K08 = 95, K09 = 106, K10 = 110, K11 = 125.

## 2. Bước 0 — tự làm tay, không tốn token

**Về `.venv` đã xóa.** `.venv` nằm trong `.gitignore` nên git không bị ảnh hưởng, và Antigravity (một IDE) không phụ thuộc vào nó. Nhưng nếu bạn xóa ở bản thật (không chỉ trong bản zip gửi đi) thì mọi gói đã cài (torch, xgboost, `pip install -e .`) đã mất, và dòng khởi động `source .venv/Scripts/activate` của plan sẽ lỗi. Cần dựng lại trước P01. Pycache trong zip là `cpython-314`, tức venv cũ dùng Python 3.14; plan (giả định A1) cảnh báo bản này có thể thiếu bản cài sẵn cho torch/xgboost/shap. Tôi không kiểm chứng được tình trạng hiện tại, nên cách chắc nhất là dựng bằng Python 3.12:

```bash
cd "/f/PT các HTTM/AITRAINING/diabetes-ai-lab"
git config core.longpaths true
py -3.12 --version            # nếu lỗi: winget install -e --id Python.Python.3.12
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip
pip install -r requirements-dev.txt && pip install -e .
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python -m pytest -q           # mong đợi: 2 passed
```

Nếu Antigravity có thiết lập giống VS Code, thêm `.venv` vào `files.watcherExclude` và `search.exclude` để nó không quét hàng chục nghìn file nhỏ. Nếu bạn dùng Python khác 3.12, sửa dòng 'Python 3.12' trong P01.

**Hai file plan nhắc tới nhưng không có trong zip:** `setup_repo.sh` (K01) và `234.pdf` (K11). Repo hiện có là nguồn sự thật nên K01 bỏ qua `setup_repo.sh`. K11 là tùy chọn; cần `234.pdf` để làm Critic C1–C4, hãy chép vào `docs/` trước khi làm P33.

Cuối cùng, lưu nội dung mục 3 thành `docs/AGENT_RULES.md` rồi commit (`git add docs/AGENT_RULES.md && git commit -m 'docs: thêm quy tắc cho agent'`). Chỉ commit khi `git status` không còn thay đổi nào khác.

## 3. Nội dung file `docs/AGENT_RULES.md`

```markdown
# Quy tắc cho agent (đọc đầu mỗi phiên)

## Khởi động mỗi phiên (Git Bash, đường dẫn luôn đặt trong dấu nháy)
cd "/f/PT các HTTM/AITRAINING/diabetes-ai-lab" && source .venv/Scripts/activate
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

## Tiết kiệm ngữ cảnh (bắt buộc)
- KHÔNG đọc cả docs/plan.md (177 KB). Chỉ đọc đúng đoạn prompt chỉ định bằng: sed -n 'A,Bp' docs/plan.md. Nếu đoạn không khớp tiêu đề bước (plan đã bị sửa), định vị lại bằng: grep -n '^#### Bước con' docs/plan.md
- Chỉ đọc file nguồn thật sự cần. Xem chữ ký bằng grep -n '^def \|^class ' <file> thay vì mở cả file. Bỏ qua danh sách 'Đọc trước khi làm' của khối nếu file chưa liên quan.
- KHÔNG liệt kê hoặc mở: .venv, .git, data/raw, data/splits, models, __pycache__, *.csv, *.npz, *.joblib. Không ls -R, không cat file lớn.
- Lệnh chạy lâu hoặc nhiều log (huấn luyện, pytest -v, nbconvert): lệnh > /tmp/k.log 2>&1; echo exit=$?; tail -n 25 /tmp/k.log. Không in log từng epoch ra khung chat.
- Không dán lại toàn văn code trong câu trả lời; không đọc lại file vừa viết trừ khi test lỗi.
- setup_repo.sh và 234.pdf không có trong repo: bỏ qua mọi bước 'đọc setup_repo.sh'. Repo hiện tại là nguồn sự thật.
- Chỉ làm ĐÚNG MỘT bước con của prompt rồi DỪNG. Không làm trước bước sau.

## Quy ước mã
pathlib; open(..., encoding='utf-8'); json.dump(..., ensure_ascii=False, indent=2); type hint đầy đủ; thông báo lỗi tiếng Việt nêu rõ cột/giá trị; matplotlib backend Agg; siêu tham số chỉ đọc từ configs/*.yaml; seed 42 qua seed.py; docstring tiếng Việt đầu mỗi file mới trong src/.

## Giao thức đánh giá
- Chọn epoch, ngưỡng, hiệu chuẩn và mô hình chỉ dựa trên VALIDATION. Tập test chỉ được nạp bởi scripts/evaluate.py (K06d).
- Accuracy luôn đi kèm baseline 'luôn đoán không'. Số lấy từ 234.pdf chỉ là mốc sanity; mọi con số kết quả phải do máy đo.
- Mọi kết quả đưa cho người dùng (API, web, báo cáo) kèm miễn trừ trách nhiệm: chỉ để học tập, không phải chẩn đoán.

## Khi gặp lỗi
Dừng, báo lỗi cùng số đo thật. KHÔNG tự sửa dữ liệu, tiêu chí hay dải chấp nhận; KHÔNG dùng kết quả test để chỉnh tham số; KHÔNG cài thêm thư viện ngoài requirements*.txt.

## Commit và tiến độ
- Commit đúng thông điệp Conventional Commits nêu trong plan, sau khi lệnh kiểm chứng cho đúng kết quả mong đợi.
- Trước khi commit, cập nhật bảng §7 của plan: chỉ sửa trong dòng (đổi [ ] thành [x], điền cột 'Kết quả đo được'), không thêm hoặc xóa dòng, để số dòng của plan không lệch.
- Cuối phiên git status --short phải rỗng.

## Báo cáo cuối phiên (tối đa 10 dòng)
File tạo/sửa; từng lệnh kiểm chứng: kết quả thật so với mong đợi; số test passed; hash commit; vấn đề hoặc điều cần người dùng quyết.
```

## 4. Khối K01 — môi trường, dữ liệu, chia tập

### P01 · K01a — Kiểm tra môi trường

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K01a rồi dừng.
Đọc: sed -n '186,214p' docs/plan.md
Làm: TẠO scripts/check_env.py; SỬA pyproject.toml (thêm pythonpath trỏ tới src và marker real_data).
Lưu ý: tôi đã dựng sẵn .venv bằng Python 3.12. KHÔNG chạy nhánh 'mv .venv', KHÔNG tạo lại venv, KHÔNG tự cài gói. Nếu core_ok hoặc torch_ok = false: dừng và báo nội dung env_check.json.
Kiểm chứng: lệnh ở dòng 'Kiểm chứng bước con'.
Mong đợi: exit=0; env_check.json có core_ok=true, csv.n_rows=253680, sha256 bắt đầu a971809d và kết thúc f9861; pytest 2 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P02 · K01b — paths, config, seed, schema

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K01b rồi dừng. K01a đã xong.
Đọc: sed -n '215,230p' docs/plan.md
Làm: SỬA paths.py, config.py, seed.py, data/schema.py (giữ nguyên TARGET, CLASSES, FEATURES), configs/data.yaml (giữ mọi khóa cũ, thêm dedup và splits_dir). Đúng chữ ký hàm trong plan.
Kiểm chứng: lệnh python -c ... trong plan, rồi pytest tests/test_schema.py.
Mong đợi: in 'diabetes-ai-lab 0.15 False 21' rồi '1 passed'.
Commit: theo dòng 'Commit trung gian'.
```

### P03 · K01c — đọc và kiểm tra dữ liệu

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K01c rồi dừng. K01a–b đã xong.
Đọc: sed -n '231,250p' docs/plan.md
Làm: SỬA data/load.py, data/validate.py, tests/conftest.py (giữ fixture sample_profile).
Lưu ý: CSV nặng 22 MB, chỉ in số đếm, không in hay dump dữ liệu.
Kiểm chứng: lệnh python -c ... trong plan.
Mong đợi: in '253680 23899 True'; file reports/metrics/data_validation.json tồn tại.
Commit: theo dòng 'Commit trung gian'.
```

### P04 · K01d — tiền xử lý, chia tập, test chống rò rỉ + tổng kết K01

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K01d rồi tổng kết khối K01. K01a–c đã xong.
Đọc: sed -n '251,282p' docs/plan.md   (bước K01d)
     sed -n '283,305p' docs/plan.md   (tiêu chí hoàn thành khối)
Làm: SỬA data/preprocess.py, data/split.py, scripts/prepare_data.py, tests/test_preprocess.py (8 test), tests/test_split_no_leakage.py (8 test).
Kiểm chứng: python scripts/prepare_data.py && python -m pytest -q tests/test_preprocess.py tests/test_split_no_leakage.py
Mong đợi: [full] train=172502 val=30442 test=50736 total=253680; [dedup] train=156250 val=27574 test=45957 total=229781; 16 passed.
Commit bước: theo dòng 'Commit trung gian'.
Cuối khối: python -m pytest -q phải 18 passed, 0 skipped; đối chiếu các tiêu chí ở đoạn 283-305; git status rỗng; commit tổng kết (nếu không còn thay đổi thì dùng --allow-empty, thân commit ghi phiên bản Python/torch/xgboost lấy từ env_check.json).
```

## 5. Khối K02 — EDA, đặc trưng, bộ chỉ số

### P05 · K02a — tiện ích đặc trưng

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K02a rồi dừng. K01 đã xong.
Đọc: sed -n '328,348p' docs/plan.md
Làm: logging_utils.py, features/feature_names.py, features/scalers.py, training/imbalance.py.
Kiểm chứng: lệnh class_weights trong plan.
Mong đợi: in [0.4989 8.8333 1.9236].
Commit: theo dòng 'Commit trung gian'.
```

### P06 · K02b — ma trận đặc trưng và bộ chỉ số

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K02b rồi dừng. K01, K02a đã xong.
Đọc: sed -n '349,370p' docs/plan.md
Làm: features/build_features.py, evaluation/metrics.py, tests/test_metrics.py, tests/test_features.py. Scaler chỉ fit trên train.
Kiểm chứng: python -m pytest -q tests/test_metrics.py tests/test_features.py
Mong đợi: 15 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P07 · K02c — EDA, từ điển dữ liệu + tổng kết K02

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K02c rồi tổng kết khối K02. K02a–b đã xong.
Đọc: sed -n '371,380p' docs/plan.md   (bước K02c)
     sed -n '381,401p' docs/plan.md   (tiêu chí hoàn thành khối)
Làm: notebooks/01_eda_brfss.ipynb (chạy xong, có output, >= 6 hình eda_*.png) và docs/data_dictionary.md.
Lưu ý: mọi con số trong kết luận EDA và từ điển phải lấy từ kết quả chạy thật, không viết tay. Thống kê liên quan nhãn chỉ tính trên tập train. Nếu nbconvert báo thiếu kernel: python -m ipykernel install --user --name python3
Kiểm chứng: lệnh nbconvert trong plan.
Cuối khối: pytest phải 33 passed; git status rỗng; commit tổng kết theo plan.
```

## 6. Khối K03 — baseline và XGBoost

### P08 · K03a — giao diện mô hình

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K03a rồi dừng. K01–K02 đã xong.
Đọc: sed -n '426,441p' docs/plan.md
Làm: models/base.py, models/baselines.py, models/xgboost_model.py, tests/test_probabilities_sum_to_one.py.
Lưu ý: nếu env_check.json có xgboost_ok=false thì dùng HistGradientBoostingClassifier và ghi backend (giả định A12).
Kiểm chứng: python -m pytest -q tests/test_probabilities_sum_to_one.py
Mong đợi: 4 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P09 · K03b — pipeline huấn luyện

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K03b rồi dừng. K03a đã xong.
Đọc: sed -n '442,461p' docs/plan.md
Làm: models/registry.py, training/train.py, scripts/train.py, tests/test_train_pipeline.py.
Lưu ý: pipeline này chỉ được nạp train và validation, không nạp test.
Kiểm chứng: python -m pytest -q tests/test_train_pipeline.py
Mong đợi: 3 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P10 · K03c — huấn luyện thật + tổng kết K03

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K03c rồi tổng kết khối K03. K03a–b đã xong.
Đọc: sed -n '462,473p' docs/plan.md   (bước K03c)
     sed -n '474,495p' docs/plan.md   (tiêu chí hoàn thành khối)
Làm: configs/train_majority.yaml, train_logreg.yaml, train_xgb.yaml; huấn luyện thật ba mô hình; ghi docs/experiment_log.md.
Lưu ý: chỉ số chỉ tính trên validation, tuyệt đối không nạp test. Log huấn luyện chuyển ra /tmp. Nếu AUC nằm ngoài dải sanity (LR 0.753–0.793, XGB 0.762–0.802) thì dừng và báo số đo cùng nghi vấn (rò rỉ, scaler, thứ tự nhãn), không sửa dải.
Kiểm chứng: vòng lặp for trong plan.
Cuối khối: pytest phải 40 passed; git status rỗng; commit tổng kết theo plan.
```

## 7. Khối K04 — MLP PyTorch

### P11 · K04a — dataset, loss, callbacks

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K04a rồi dừng. K01–K03 đã xong và env_check.json có torch_ok=true (nếu không: dừng và báo).
Đọc: sed -n '519,529p' docs/plan.md
Làm: data/dataset.py, training/losses.py, training/callbacks.py.
Kiểm chứng: lệnh python -c ... trong plan (in tên loss, best_epoch, kết quả step).
Commit: theo dòng 'Commit trung gian'.
```

### P12 · K04b — MLP và trainer

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K04b rồi dừng. K04a đã xong.
Đọc: sed -n '530,556p' docs/plan.md
Làm: models/mlp.py, training/trainer.py, tests/test_mlp.py.
Lưu ý: chỉ CPU, num_workers=0; test dùng dữ liệu rất nhỏ và ít epoch để chạy trong vài giây. Chọn epoch chỉ bằng AUC validation.
Kiểm chứng: python -m pytest -q tests/test_mlp.py
Mong đợi: 4 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P13 · K04c — đăng ký, cấu hình, test trainer

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K04c rồi dừng. K04a–b đã xong.
Đọc: sed -n '557,567p' docs/plan.md
Làm: SỬA models/registry.py (đăng ký mlp), configs/train_mlp.yaml, TẠO tests/test_trainer.py.
Kiểm chứng: lệnh ở dòng 'Kiểm chứng bước con' (pytest test_trainer rồi build_model('mlp', ...)).
Mong đợi: test_trainer passed; build_model in ra tên mô hình mlp.
Commit: theo dòng 'Commit trung gian'.
```

### P14 · K04d — huấn luyện MLP thật + tổng kết K04

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K04d rồi tổng kết khối K04. K04a–c đã xong.
Đọc: sed -n '568,580p' docs/plan.md   (bước K04d)
     sed -n '581,601p' docs/plan.md   (tiêu chí hoàn thành khối)
Làm: huấn luyện MLP thật trên CPU (tối đa 10 phút/lần), chạy lần hai để kiểm tra tái lập, ghi docs/experiment_log.md, xóa run _repeat sau khi so sánh.
Lưu ý: chạy lệnh nền với log ra /tmp, chỉ tail vài dòng cuối, kiểm tra thưa (cách nhau vài phút). Không đụng test.
Mong đợi: AUC val MLP trong [0.762, 0.802]; tái lập chênh <= 1e-4.
Cuối khối: pytest phải 48 passed; git status rỗng; commit tổng kết theo plan.
```

## 8. Khối K05 — hiệu chuẩn, ngưỡng, khoảng tin cậy, chọn mô hình

### P15 · K05a + K05b — khoảng tin cậy và hiệu chuẩn

```
Làm theo docs/AGENT_RULES.md. Làm hai bước K05a rồi K05b (mỗi bước một commit), rồi dừng. K01–K04 đã xong.
Đọc: sed -n '623,641p' docs/plan.md   (K05a)
     sed -n '642,663p' docs/plan.md   (K05b)
Làm: evaluation/confidence_intervals.py + tests/test_confidence_intervals.py; evaluation/calibration.py + tests/test_calibration.py.
Kiểm chứng: pytest từng file test theo plan.
Mong đợi: 4 passed, rồi 4 passed.
Commit: theo dòng 'Commit trung gian' của từng bước.
```

### P16 · K05c — ngưỡng

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K05c rồi dừng. K05a–b đã xong.
Đọc: sed -n '664,674p' docs/plan.md
Làm: evaluation/thresholds.py + tests/test_thresholds.py. Ngưỡng chỉ chọn trên validation.
Kiểm chứng: python -m pytest -q tests/test_thresholds.py
Mong đợi: 4 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P17 · K05d — chạy thật, chọn mô hình + tổng kết K05

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K05d rồi tổng kết khối K05. K05a–c đã xong.
Đọc: sed -n '675,684p' docs/plan.md   (bước K05d)
     sed -n '685,706p' docs/plan.md   (tiêu chí hoàn thành khối)
     sed -n '97,116p' docs/plan.md    (giả định A6, A7: quy tắc chọn mô hình và xác suất hiển thị)
Làm: configs/evaluate.yaml, scripts/calibrate_and_threshold.py; chạy thật cho ba run *_multiclass_full; ghi docs/experiment_log.md (ECE trước → sau, mô hình tốt nhất).
Lưu ý: hiệu chuẩn, ngưỡng và chọn mô hình chỉ dùng validation, không nạp test. Nếu ECE hoặc macro-F1 không đạt tiêu chí thì dừng và báo số đo, không nới tiêu chí.
Mong đợi: ECE macro (out-of-fold) <= 0.03 và thấp hơn trước hiệu chuẩn; in ra mô hình tốt nhất theo quy tắc A6.
Cuối khối: pytest phải 60 passed; git status rỗng; commit tổng kết theo plan.
```

## 9. Khối K06 — công bằng, đánh giá test một lần

### P18 · K06a — công bằng theo nhóm

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K06a rồi dừng. K01–K05 đã xong.
Đọc: sed -n '728,748p' docs/plan.md
Làm: evaluation/fairness.py, tests/test_fairness.py, SỬA configs/evaluate.yaml (cấu hình nhóm).
Kiểm chứng: python -m pytest -q tests/test_fairness.py
Mong đợi: 4 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P19 · K06b — biểu đồ và báo cáo số liệu

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K06b rồi dừng. K06a đã xong.
Đọc: sed -n '749,761p' docs/plan.md
Làm: evaluation/plots.py, evaluation/report.py, tests/test_report_plots.py.
Kiểm chứng: python -m pytest -q tests/test_report_plots.py
Mong đợi: 2 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P20 · K06c — huấn luyện đủ các biến thể (không viết code mới)

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K06c rồi dừng. K06a–b đã xong.
Đọc: sed -n '762,777p' docs/plan.md
Làm: chỉ chạy lệnh theo plan để huấn luyện và hiệu chuẩn các biến thể nhị phân và dedup trên validation. Không sửa code.
Lưu ý: bước này chạy lâu (có MLP). Chạy nền, log ra /tmp, chỉ tail vài dòng cuối, kiểm tra thưa. Không nạp test. Bạn có thể tự chạy các lệnh này trong terminal rồi bảo agent chỉ kiểm chứng và commit.
Kiểm chứng: lệnh python -c in model_selection.json trong plan.
Commit: theo dòng 'Commit trung gian'.
```

### P21 · K06d — đánh giá test đúng một lần + tổng kết K06

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K06d rồi tổng kết khối K06. K06a–c đã xong.
Đọc: sed -n '778,799p' docs/plan.md   (bước K06d)
     sed -n '800,822p' docs/plan.md   (tiêu chí hoàn thành khối)
Làm: scripts/evaluate.py, tests/test_evaluate_guard.py, ghi docs/experiment_log.md.
QUAN TRỌNG về tập test:
1. Trước hết chạy: test -f reports/metrics/test_lock.json && echo DA_CO_KHOA. Nếu đã có khóa: KHÔNG chạy lại evaluate.py --split test, chỉ báo cáo.
2. Gỡ lỗi script chỉ bằng: python scripts/evaluate.py --split val. Chỉ chạy --split test đúng MỘT lần, khi script đã ổn trên validation và pytest đang xanh.
3. Cụm include_test=True chỉ được xuất hiện trong scripts/evaluate.py (kể cả docstring/bình luận của file khác), nếu không test guard sẽ lỗi.
4. Tuyệt đối không chỉnh siêu tham số, ngưỡng hay lựa chọn mô hình dựa trên kết quả test. Nếu kết quả ngoài dải sanity: dừng và báo số đo.
Log ra /tmp (bootstrap 1000 lần khá lâu).
Kiểm chứng: hai lệnh ở dòng 'Kiểm chứng bước con'. Mong đợi: 3 passed và --split val không ghi khóa; rồi lần chạy test đầu exit=0, lần hai exit=2, test_lock.json có đúng 1 mục.
Cuối khối: pytest phải 69 passed; git status rỗng; commit tổng kết theo plan.
```

## 10. Khối K07 — giải thích

### P22 · K07a — tác động yếu tố và what-if

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K07a rồi dừng. K01–K06 đã xong.
Đọc: sed -n '843,865p' docs/plan.md
Làm: SỬA configs/features.yaml (mục what_if), explain/what_if.py, tests/test_what_if.py.
Lưu ý: what-if chỉ trên 6 yếu tố cấu hình và luôn kèm ghi chú 'tương quan, không phải nhân quả'.
Kiểm chứng: python -m pytest -q tests/test_what_if.py
Mong đợi: 4 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P23 · K07b + K07c — hoán vị, SHAP, biểu đồ + tổng kết K07

```
Làm theo docs/AGENT_RULES.md. Làm K07b rồi K07c (mỗi bước một commit), rồi tổng kết khối K07. K07a đã xong.
Đọc: sed -n '866,875p' docs/plan.md   (K07b)
     sed -n '876,885p' docs/plan.md   (K07c)
     sed -n '886,906p' docs/plan.md   (tiêu chí hoàn thành khối)
Làm: explain/permutation.py, explain/shap_explain.py, tests/test_explain.py (4 test); SỬA evaluation/plots.py, TẠO scripts/explain_global.py.
Lưu ý: giải thích toàn cục tính trên VALIDATION, không nạp test. SHAP là tùy chọn: nếu shap không cài được thì bỏ qua và ghi chú.
Kiểm chứng: lệnh ở dòng 'Kiểm chứng bước con' của từng bước. Mong đợi: 4 passed; permutation_importance_*.json có 21 mục.
Cuối khối: pytest phải 77 passed; git status rỗng; commit tổng kết theo plan.
```

## 11. Khối K08 — luật an toàn, predictor, xuất mô hình

### P24 · K08a — luật an toàn và mức nguy cơ

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K08a rồi dừng. K01–K07 đã xong.
Đọc: sed -n '930,958p' docs/plan.md
Làm: safety/rules.py, safety/risk_level.py, safety/disclaimers.py, SỬA configs/safety_rules.yaml, tests/test_safety_rules.py.
Lưu ý: luật đọc từ YAML; KHÔNG dùng eval trên chuỗi luật chưa qua kiểm tra bằng ast.
Kiểm chứng: python -m pytest -q tests/test_safety_rules.py
Mong đợi: 10 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P25 · K08b — lược đồ và gói xuất mô hình

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K08b rồi dừng. K08a đã xong.
Đọc: sed -n '959,972p' docs/plan.md
Làm: inference/schemas.py, inference/export.py, scripts/export_model.py.
Lưu ý: models/best_model.joblib bị .gitignore, đừng ép commit file này.
Kiểm chứng: lệnh export_model + python -c in run_name, task, số đặc trưng, dung lượng.
Mong đợi: file models/best_model.joblib tạo được; feature_order có 21 phần tử.
Commit: theo dòng 'Commit trung gian'.
```

### P26 · K08c — Predictor + tổng kết K08

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K08c rồi tổng kết khối K08. K08a–b đã xong.
Đọc: sed -n '973,982p' docs/plan.md   (bước K08c)
     sed -n '983,1003p' docs/plan.md  (tiêu chí hoàn thành khối)
Làm: SỬA inference/predictor.py, tests/conftest.py, tests/test_predictor.py.
Lưu ý: mọi phản hồi của Predictor phải có miễn trừ trách nhiệm và lời khuyên xét nghiệm xác nhận ở mọi mức nguy cơ.
Kiểm chứng: python -m pytest -q tests/test_predictor.py
Mong đợi: 8 passed.
Cuối khối: pytest phải 95 passed; git status rỗng; commit tổng kết theo plan.
```

## 12. Khối K09 — API và demo web

### P27 · K09a + K09b — lõi API và test API

```
Làm theo docs/AGENT_RULES.md. Làm K09a rồi K09b (mỗi bước một commit), rồi dừng. K01–K08 đã xong và đã có models/best_model.joblib.
Đọc: sed -n '1025,1037p' docs/plan.md   (K09a)
     sed -n '1038,1046p' docs/plan.md   (K09b)
Làm: serving/deps.py, routes.py, app.py; configs/serve.yaml; tests/test_api_predict.py.
Lưu ý: giữ nguyên /health trả đúng {"status": "ok"} (không thêm trường mới; trạng thái mô hình đặt ở /model-info). Không lưu dữ liệu người dùng.
Kiểm chứng: tests/test_api_health.py vẫn 1 passed; tests/test_api_predict.py 8 passed.
Commit: theo dòng 'Commit trung gian' của từng bước.
```

### P28 · K09c — demo web + tổng kết K09

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K09c rồi tổng kết khối K09. K09a–b đã xong.
Đọc: sed -n '1047,1057p' docs/plan.md   (bước K09c)
     sed -n '1058,1096p' docs/plan.md   (tiêu chí hoàn thành khối, gồm smoke test)
Làm: app/index.html, app/static/css/style.css, app/static/js/app.js, tests/test_demo_static.py.
Lưu ý: trang web hiển thị miễn trừ trách nhiệm. Smoke test: chạy uvicorn nền trên cổng 8200, gọi thử, rồi TẮT server khi xong. Nếu cổng bị chiếm: netstat -ano | findstr :8200.
Kiểm chứng: python -m pytest -q tests/test_demo_static.py (3 passed) + smoke test theo plan.
Cuối khối: pytest phải 106 passed; git status rỗng; commit tổng kết theo plan.
```

## 13. Khối K10 — tài liệu, báo cáo tự sinh, chạy lại toàn pipeline

### P29 · K10a — công cụ sinh báo cáo và run\_all.sh

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K10a rồi dừng. K01–K09 đã xong.
Đọc: sed -n '1122,1132p' docs/plan.md
Làm: scripts/make_reports.py, scripts/run_all.sh, SỬA Makefile, SỬA .gitattributes (thêm quy ước *.sh text eol=lf).
Lưu ý: Git Bash thường không có make; đường chạy chính là bash scripts/run_all.sh.
Kiểm chứng: lệnh bash -n ... && make_reports.py --help && git check-attr trong plan.
Mong đợi: không lỗi cú pháp, in trợ giúp, 'eol: lf'.
Commit: theo dòng 'Commit trung gian'.
```

### P30 · K10b — báo cáo trong reports/

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K10b rồi dừng. K10a đã xong.
Đọc: sed -n '1133,1142p' docs/plan.md
Làm: reports/model_card.md, reports/fairness_report.md, reports/final_report.md.
Lưu ý: mọi con số đến từ scripts/make_reports.py (khối AUTO), không chép tay. Phần diễn giải phải trung thực về hạn chế (nhóm tiền tiểu đường khó phân biệt, accuracy thấp hơn baseline, chỉ là tương quan).
Kiểm chứng: python scripts/make_reports.py && python scripts/make_reports.py --check
Mong đợi: lần hai thoát 0; grep -c 'AUTO:' reports/model_card.md >= 7.
Commit: theo dòng 'Commit trung gian'.
```

### P31 · K10c — tài liệu docs/ và README

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K10c rồi dừng. K10a–b đã xong.
Đọc: sed -n '1143,1154p' docs/plan.md
Làm: docs/architecture.md, docs/safety_and_ethics.md, docs/references.md, README.md, docs/experiment_log.md.
Kiểm chứng: lệnh grep -c miễn trừ trách nhiệm trong plan.
Mong đợi: mỗi file README.md, docs/safety_and_ethics.md, reports/model_card.md có >= 1.
Commit: theo dòng 'Commit trung gian'.
```

### P32 · K10d — test tài liệu, chạy lại toàn pipeline + tổng kết K10

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K10d rồi tổng kết khối K10. K10a–c đã xong.
Đọc: sed -n '1155,1190p' docs/plan.md   (bước K10d)
     sed -n '1191,1213p' docs/plan.md   (tiêu chí hoàn thành khối)
     sed -n '1122,1132p' docs/plan.md   (đặc tả run_all.sh và --clean: đọc để biết lệnh xóa và tạo lại những gì)
Làm: tests/test_docs_disclaimer.py (4 test); rồi chạy lại sạch toàn pipeline ĐÚNG MỘT LẦN để kiểm tra tái lập.
Trước khi chạy: git status phải rỗng; cp reports/metrics/test_evaluation.json /tmp/test_eval_before.json.
Lưu ý: lệnh chạy tối đa khoảng 60 phút. Chạy nền, log ra /tmp, kiểm tra thưa (cách nhau vài phút), không in log từng dòng. Đây là phát lại để kiểm tra tái lập, không phải để chỉnh tham số. Nếu làm hỏng phân nửa chừng, dừng và báo cáo, không tự chạy lại.
Kiểm chứng: hai lệnh ở dòng 'Kiểm chứng bước con'.
Mong đợi: 4 passed; rồi in 'số lần đánh giá = 2 | lần 2 forced = True' với sai khác tối đa <= 1e-3.
Cuối khối: pytest phải 110 passed; git status rỗng; commit tổng kết theo plan.
```

## 14. Khối K11 — tùy chọn

Chỉ làm khi K10 đã đạt toàn bộ tiêu chí và đã commit. K11 không được làm thay đổi kết quả K01–K10. Plan nêu Critic C1–C4 theo `234.pdf` (mục 4.2, 4.4, 5.6.1); nếu chưa chép `234.pdf` vào `docs/`, thêm vào P33 dòng: 'Nếu không có docs/234.pdf, hỏi tôi trước, đừng tự bịa C1–C4.'

### P33 · K11a — critic, kiểm định chéo, tuning

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K11a rồi dừng. K01–K10 đã xong và đã commit.
Đọc: sed -n '1241,1251p' docs/plan.md
Làm: safety/critic.py, training/cross_validation.py, training/tuning.py, tests/test_critic.py, tests/test_cv_tuning.py.
Lưu ý: tuning chỉ trên validation. Không đổi kết quả hay file của K01–K10 ngoài những gì bước này nêu.
Kiểm chứng: python -m pytest -q tests/test_critic.py tests/test_cv_tuning.py
Mong đợi: 7 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P34 · K11b — tri thức BM25 và Graph-RAG tối giản

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K11b rồi dừng. K11a đã xong.
Đọc: sed -n '1252,1263p' docs/plan.md
Làm: knowledge/build_kb.py, knowledge/bm25.py, knowledge/graph_rag.py, tests/test_knowledge.py, knowledge_base/README.md.
Kiểm chứng: python -m pytest -q tests/test_knowledge.py
Mong đợi: 4 passed.
Commit: theo dòng 'Commit trung gian'.
```

### P35 · K11c — agents + tổng kết K11

```
Làm theo docs/AGENT_RULES.md. Chỉ làm bước K11c rồi tổng kết khối K11. K11a–b đã xong.
Đọc: sed -n '1264,1273p' docs/plan.md   (bước K11c)
     sed -n '1274,1292p' docs/plan.md   (tiêu chí hoàn thành khối)
Làm: agents/orchestrator.py, risk_agent.py, knowledge_agent.py, safety_agent.py, tests/test_agents.py.
Kiểm chứng: python -m pytest -q tests/test_agents.py
Mong đợi: 4 passed.
Cuối khối: pytest phải 125 passed; git status rỗng; commit tổng kết theo plan.
```

## 15. Prompt TIẾP TỤC (khi agent bị ngắt giữa chừng)

Mở cuộc trò chuyện mới, thay `<Kxxy>` và hai số dòng `A,B` bằng bước đang dở (lấy từ prompt gốc):

```
Phiên trước bị ngắt giữa chừng khi làm bước <Kxxy>. Làm theo docs/AGENT_RULES.md.
1) Chạy: git status --short; git log --oneline -n 5; python -m pytest -q 2>&1 | tail -n 15
2) Đọc đúng đoạn plan của bước này: sed -n 'A,Bp' docs/plan.md
3) So danh sách file của bước với file đã có (kiểm tra từng file, không ls -R). File nào đã có và chạy đúng thì giữ nguyên, KHÔNG viết lại từ đầu; chỉ sửa file thiếu hoặc hỏng.
4) Làm nốt, chạy lệnh kiểm chứng, commit đúng thông điệp trong plan, rồi dừng và báo cáo.
Ngoại lệ: nếu bước là K06d và reports/metrics/test_lock.json đã tồn tại thì tuyệt đối không chạy lại evaluate.py --split test.
```

Nếu một prompt vẫn quá nặng, tách đôi nó: ở cuộc 1 thêm dòng 'Chỉ làm các file: X, Y; chưa commit; báo cáo', rồi ở cuộc 2 dùng prompt TIẾP TỤC để làm phần còn lại, kiểm chứng và commit.
