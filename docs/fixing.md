# fixing.md — Kế hoạch sửa lỗi và bổ sung cho `diabetes-ai-lab`

> Tài liệu giao việc cho **agent thực thi**. Mỗi mục `Fxx` tự đủ thông tin: bằng chứng, file cần sửa, nội dung trước/sau, test cần thêm, lệnh kiểm chứng, commit. Agent không cần nhớ cuộc trò chuyện nào khác.
> Dự án chỉ phục vụ học tập, không phải công cụ chẩn đoán y khoa.

---

## 0. Cách dùng và quy tắc bắt buộc

### 0.1 Khởi động mỗi phiên (Git Bash, Windows)

Đường dẫn gốc dự án (theo `reports/metrics/export_summary.json`) có dấu tiếng Việt và khoảng trắng, nên luôn đặt trong dấu nháy:

```bash
cd "/f/PT các HTTM/AITRAINING/diabetes-ai-lab" && source .venv/Scripts/activate
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
```

Nếu repo nằm ở chỗ khác thì `cd` tới thư mục chứa `pyproject.toml`. Python của người dùng là 3.14.5 (xem `reports/metrics/env_check.json`); CI dùng 3.11.

### 0.2 Quy tắc cứng (không được vi phạm)

1. **Không chạm vào kết quả test đã khóa.** Không sửa tay, không xóa, không tạo lại: `reports/metrics/test_lock.json`, `reports/metrics/test_evaluation.json`. Không chạy `python scripts/evaluate.py` không kèm `--split val`. Không chạy `run_all.sh --replay-test` và không dùng `--clean`.
2. **Không huấn luyện lại, không đổi seed, không đổi `configs/*.yaml` ngoài phần được nêu rõ.** Mọi số đo hiện có (AUC test MLP 0,7858; XGBoost nhị phân 0,8244…) phải giữ nguyên. Riêng F10 có chạy lại `scripts/export_model.py` (chỉ đóng gói lại, không huấn luyện).
3. **Tập test chỉ được nạp bởi `scripts/evaluate.py`.** Sau mọi thay đổi, lệnh sau chỉ được trả về dòng của `scripts/evaluate.py`:
   ```bash
   grep -rn "include_test=True" src scripts
   ```
4. **Quy ước mã** (từ `docs/plan.md`): `pathlib`; mở file luôn `encoding="utf-8"`; `json.dump(..., ensure_ascii=False, indent=2)`; type hint đầy đủ; thông báo lỗi tiếng Việt nêu rõ trường/giá trị; docstring tiếng Việt đầu file; matplotlib dùng backend `Agg`.
5. **Giữ tương thích ngược:** không đổi tên khóa JSON trả về của `/predict`, `/explain`, `/health` (`/health` luôn trả đúng `{"status": "ok"}`); không đổi `schema_version` của bundle (hiện là `1`).
6. **File có kết thúc dòng CRLF** (ví dụ `reports/final_report.md`, các `*.json` sinh trên Windows): khi sửa tài liệu bằng script, đọc/ghi dạng bytes như hàm `sub()` ở Phụ lục A để không làm đổi toàn bộ file trong `git diff`.
7. **Khi một bước thất bại:** dừng lại, báo lỗi và số liệu đo được; không tự sửa dữ liệu, tiêu chí hay dùng test để chỉnh tham số.
8. **Commit:** Conventional Commits (`fix|feat|test|docs|chore|ci(<scope>): ...`). Mỗi mục `Fxx` một commit (hoặc một commit/bước con nếu có ghi). `git status --short` phải rỗng sau mỗi mục.

### 0.3 Bước nền B0 (làm trước mọi mục)

```bash
git status --short                       # kỳ vọng: rỗng (nếu không, dừng và báo)
python -m pytest -q 2>&1 | tail -15      # GHI LẠI số passed/failed/skipped trước khi sửa
python -m ruff check src tests 2>&1 | tail -15   # GHI LẠI số lỗi lint (nếu ruff chưa cài: pip install ruff)
```

Ghi hai kết quả này vào nội dung commit đầu tiên hoặc báo lại cho người dùng. **Người lập kế hoạch chưa chạy được `pytest`, `ruff` hay `uvicorn`** (môi trường lập kế hoạch không có mạng để cài thư viện); mọi kết luận dưới đây rút ra từ **đọc mã, đọc số liệu `reports/metrics/*.json` và nạp thử `models/best_model.joblib` bằng bộ nạp tùy biến**. Vì vậy mỗi mục đều có lệnh kiểm chứng để agent tự xác nhận.

### 0.4 Thứ tự thực hiện

```
B0 → F01 → F02 → F03 → F04          (nhóm P0: lỗi chức năng, làm tuần tự)
   → F05 → F06 → F07 → F08 → F09 → F10   (nhóm P1)
   → F11 → F12 → F13 → F14 → F15         (nhóm P2, có thể làm theo thứ tự bất kỳ)
   → Mục 6 (nghiệm thu cuối)
```

F02 phải làm sau F01 (test hợp đồng UI ở F01 dùng cấu hình `what_if`). F06 phải làm trước F04 nếu muốn test `.env`; nhưng F04 tự đủ nên có thể làm theo thứ tự trên. F08 (`/consult`) phụ thuộc F03 và F04.

---

## 1. Kết quả đối chiếu `plan.md` ↔ mã hiện tại

Ký hiệu: ✅ đạt (đã kiểm bằng số liệu trong repo), ❌ chưa đạt, ⚠️ đạt một phần / cần xác nhận.

| Khối | Tiêu chí chính trong plan | Hiện trạng | Ghi chú |
|---|---|---|---|
| K01 | Chia 172.502/30.442/50.736 và 156.250/27.574/45.957, giao nhau = 0 | ✅ | khớp `data/splits/*/manifest.json` |
| K02 | `ExperimentData`, scaler, metrics, **notebook EDA ≥ 14 ô, ≥ 6 hình `eda_*.png`, `data_dictionary.md` đủ số** | ❌ | `notebooks/01_eda_brfss.ipynb` = 195 byte (stub), không có `reports/figures/eda_*.png`, `docs/data_dictionary.md` = 129 byte (stub) → **F05** |
| K03 | AUC val LR ∈ [0,753; 0,793]; XGB ∈ [0,762; 0,802]; AUC lớp 1 ∈ [0,65; 0,73] | ✅ | LR 0,7722 (lớp 1: 0,6857); XGB 0,7807 (0,6912) |
| K04 | AUC val MLP ∈ [0,762; 0,802]; `history` đúng 14 mục | ✅ | MLP 0,7840 (lớp 1: 0,7030); 14 epoch |
| K05 | ECE sau hiệu chuẩn ≤ 0,03 và thấp hơn trước; macro-F1 scaled ≥ argmax | ✅ | MLP full: ECE 0,0046 → 0,0016; F1 0,3941 → 0,4699 |
| K06 | n=50.736 / 45.957; baseline 0,8424; AUC test đúng khoảng; CI ≤ 0,02; PPV nhóm cao ≥ 20,9% | ✅ | n đúng; LR 0,7719 / XGB 0,7873 / MLP 0,7858; CI rộng 0,0127; PPV cao 38,0% |
| K06 | `test_lock.json` | ✅ | 2 mục (lần 1 thường, lần 2 `forced`, `reason="run_all replay"`) — đúng trạng thái sau K10 |
| K07 | 21 mục độ quan trọng, sắp giảm dần | ✅ | top 5: GenHlth, BMI, Age, HighChol, HighBP |
| K08 | `best_model.joblib` < 50 MB, có R1, luôn có "xét nghiệm xác nhận" | ✅ | 62.048 byte |
| K09 | API 200/422/503; trang `/app/` có dải miễn trừ | ⚠️ | API đúng, nhưng **giao diện what-if gửi trường không được phép → server trả 500** → **F01, F02** |
| K10 | tài liệu, `make_reports.py --check`, `run_all.sh` | ⚠️ | README/architecture/final_report có số và mô tả sai (F07); `run_all.sh` chưa gọi `--check` (F13) |
| K11 (tùy chọn) | `Orchestrator.consult` kết thúc bằng `final`, có `disclaimer` | ⚠️ | có mã và test, nhưng **chưa nối vào API/UI** (F08) và **`SafetyAgent` đọc sai khóa xác suất** (F03) |
| Hạ tầng | CI/Docker (plan A16: bỏ qua, nhưng repo đã có) | ❌ | `Makefile` hỏng, CI sẽ fail trên bản clone sạch, Docker cài cả `torch` (F04, F09) |

**Số test:** plan nêu 110 (K10) và 125 (K11) test; thư mục `tests/` hiện có **131 hàm `def test_`** (chưa tính tham số hóa). Không coi đây là lỗi; ở Mục 6 chỉ cần ghi lại số thực đo.

---

## 2. Bảng tổng hợp lỗi và hạng mục bổ sung

| ID | Mức | Loại | Tóm tắt | File chính |
|---|---|---|---|---|
| F01 | **P0** | Lỗi chức năng | Giao diện what-if gửi `HighBP`, `HighChol` (không nằm trong 6 khóa được phép) → `/explain` lỗi 500, giao diện nuốt lỗi, panel không cập nhật | `app/index.html`, `app/static/js/app.js` |
| F02 | **P0** | Lỗi chức năng | `/explain` không bắt `ValueError` → 500 thay vì 422; giá trị `NaN` lọt qua kiểm tra miền | `serving/routes.py`, `inference/schemas.py`, `explain/what_if.py` |
| F03 | **P0** | Lỗi logic | `SafetyAgent` tìm khóa `"Tiểu đường"` (hoa) trong khi tên lớp là `"tiểu đường"` → luật R10 dùng `risk_index` thay vì P(tiểu đường) | `agents/safety_agent.py` |
| F04 | **P0** | Lỗi kiểm thử/CI | `tests/test_agents.py` gọi `Orchestrator()` → nạp `models/best_model.joblib` thật (bị `.gitignore` bỏ qua) → fail trên bản clone sạch/CI | `tests/test_agents.py` |
| F05 | P1 | Thiếu so với plan | K02c chưa làm: notebook EDA, hình `eda_*.png`, từ điển dữ liệu | `notebooks/01_eda_brfss.ipynb`, `docs/data_dictionary.md` |
| F06 | P1 | Lỗi cấu hình | `MODEL_PATH` trong `.env` bị bỏ qua (code đọc `os.environ` trực tiếp, không qua `get_env`); 4 biến trong `.env.example` không được dùng | `serving/deps.py`, `inference/predictor.py`, `.env.example` |
| F07 | P1 | Tài liệu sai | README/architecture/final_report lệch với code và số đo | `README.md`, `docs/architecture.md`, `reports/final_report.md` |
| F08 | P1 | Thiếu tính năng | Hệ đa agent (`Orchestrator`, Graph-RAG) chưa có endpoint/giao diện | `serving/*`, `inference/schemas.py`, `app/*` |
| F09 | P1 | Hạ tầng | `Makefile` target `all` sai cú pháp; Docker/CI cài cả `torch`+`shap`; compose bắt buộc có `.env`; CI không cache | `Makefile`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.github/workflows/ci.yml`, `requirements-serve.txt` |
| F10 | P1 | Độ bền | Bundle không ghi phiên bản thư viện (joblib/pickle nhạy phiên bản scikit-learn); `export_summary.json` chứa đường dẫn tuyệt đối Windows | `inference/export.py`, `inference/predictor.py`, `scripts/export_model.py` |
| F11 | P2 | Bảo mật nhẹ | `innerHTML` với dữ liệu từ máy chủ; CORS cho phép origin `"null"` | `app/static/js/app.js`, `serving/app.py`, `configs/serve.yaml` |
| F12 | P2 | Hiệu năng/vận hành | `Predictor.explain` đọc `features.yaml` mỗi request; `/health` không phản ánh trạng thái nạp mô hình | `inference/predictor.py`, `serving/app.py` |
| F13 | P2 | Quy trình | `run_all.sh` chưa gọi `make_reports.py --check` (plan K10 yêu cầu) | `scripts/run_all.sh` |
| F14 | P2 | Tái lập | `env_check.json` không ghi phiên bản gói | `scripts/check_env.py` |
| F15 | P2 | Giao tiếp kết quả | `label` có thể là "tiền tiểu đường" dù xác suất lớp đó rất thấp (hệ số ngưỡng 17,98); ~50% người dùng bị xếp ≥ "trung bình" — chưa được nêu trong tài liệu | `inference/schemas.py`, `reports/model_card.md`, `reports/final_report.md` |

### Những điểm đã kiểm tra và **không** phải lỗi (agent không cần sửa)

- AUC test XGBoost đa lớp (0,7873) cao hơn MLP (0,7858) nhưng mô hình được chọn là MLP: đúng quy tắc A6 (chọn trên **validation**, ngưỡng đơn giản 0,003). Không đổi.
- Brier sau hiệu chuẩn của MLP đa lớp tăng rất nhẹ (0,22207 → 0,22215) trong khi ECE giảm mạnh: chấp nhận được, không cần xử lý.
- Hiệu chuẩn và ngưỡng cùng chọn trên validation (đã ghi chú trong `postproc_*.json` và dùng out-of-fold).
- Khoảng 12,7% dòng val/test là bản sao hệt dòng train (biến thể `full`): đã được báo cáo bằng "tập con không bản sinh đôi" (AUC MLP 0,7750) và biến thể `dedup`.
- Bài toán nhị phân: mô hình tốt nhất là XGBoost nhưng ứng dụng chỉ phục vụ bundle đa lớp (MLP) — đúng thiết kế.

---

## 3. Nhóm P0 — lỗi chức năng (làm trước)

### F01 — Giao diện what-if gửi trường không được phép → `/explain` lỗi 500

**Bằng chứng (đã xác minh):**
- `app/static/js/app.js`, hàm `triggerWhatIfUpdate`, gửi `what_if_changes` gồm `BMI, HighBP, HighChol, Smoker, PhysActivity, Fruits`.
- `Predictor.explain` chỉ cho phép các khóa trong `what_if` của gói mô hình. Nạp thật `models/best_model.joblib` cho thấy khóa được phép là đúng 6: `BMI, PhysActivity, Smoker, HvyAlcoholConsump, Fruits, Veggies` (trùng khối `what_if` trong `configs/features.yaml`). `HighBP`, `HighChol` **không** nằm trong danh sách (chúng là bệnh nền, không phải lối sống thay đổi được).
- `explain/what_if.py::what_if_custom` ném `ValueError` khi có khóa lạ; `serving/routes.py` không bắt → HTTP 500. `callExplain` trong `app.js` gặp `!res.ok` thì `return` im lặng → thanh trượt BMI và ô bật/tắt **không bao giờ cập nhật được bảng what-if** sau lần tải đầu.
- Test hiện có chỉ gửi `{"BMI": 22.0}` (`tests/test_api_predict.py:128`) nên không phát hiện.

**Hướng sửa (giữ nguyên cấu hình & mô hình):** sửa giao diện cho khớp với 5 yếu tố lối sống được phép (thay `HighBP`, `HighChol` bằng `Veggies`, `HvyAlcoholConsump`), và hiển thị lỗi thay vì nuốt lỗi. Server cũng được làm chắc hơn ở F02.

**Bước F01.1 — `app/index.html`** (khối `whatif-toggles`, quanh dòng 320–326). Thay

```html
            <div class="whatif-toggles">
              <label class="toggle-item"><input type="checkbox" id="whatif-HighBP"> Huyết áp cao</label>
              <label class="toggle-item"><input type="checkbox" id="whatif-HighChol"> Mỡ máu cao</label>
              <label class="toggle-item"><input type="checkbox" id="whatif-Smoker"> Hút thuốc</label>
              <label class="toggle-item"><input type="checkbox" id="whatif-PhysActivity"> Vận động thể chất</label>
              <label class="toggle-item"><input type="checkbox" id="whatif-Fruits"> Ăn trái cây</label>
            </div>
```

bằng

```html
            <div class="whatif-toggles">
              <label class="toggle-item"><input type="checkbox" id="whatif-Smoker"> Hút thuốc</label>
              <label class="toggle-item"><input type="checkbox" id="whatif-PhysActivity"> Vận động thể chất</label>
              <label class="toggle-item"><input type="checkbox" id="whatif-Fruits"> Ăn trái cây</label>
              <label class="toggle-item"><input type="checkbox" id="whatif-Veggies"> Ăn rau</label>
              <label class="toggle-item"><input type="checkbox" id="whatif-HvyAlcoholConsump"> Uống rượu bia nhiều</label>
            </div>
```

Không cần sửa CSS (`.whatif-toggles`, `.toggle-item` dùng chung).

**Bước F01.2 — `app/static/js/app.js`**, ba chỗ (dùng `str_replace` hoặc hàm `sub()` ở Phụ lục A; mỗi đoạn `old` phải khớp đúng 1 lần):

(a) Trong `initWhatIfControls`, thay

```javascript
  // 5 yếu tố có thể thay đổi
  document.getElementById("whatif-HighBP").checked = profile.HighBP === 1;
  document.getElementById("whatif-HighChol").checked = profile.HighChol === 1;
  document.getElementById("whatif-Smoker").checked = profile.Smoker === 1;
  document.getElementById("whatif-PhysActivity").checked = profile.PhysActivity === 1;
  document.getElementById("whatif-Fruits").checked = profile.Fruits === 1;
```

bằng

```javascript
  // 5 yếu tố lối sống có thể thay đổi (khớp khối `what_if` trong configs/features.yaml)
  document.getElementById("whatif-Smoker").checked = profile.Smoker === 1;
  document.getElementById("whatif-PhysActivity").checked = profile.PhysActivity === 1;
  document.getElementById("whatif-Fruits").checked = profile.Fruits === 1;
  document.getElementById("whatif-Veggies").checked = profile.Veggies === 1;
  document.getElementById("whatif-HvyAlcoholConsump").checked = profile.HvyAlcoholConsump === 1;
```

(b) Trong `triggerWhatIfUpdate`, thay

```javascript
    const whatIfChanges = {
      BMI: parseFloat(document.getElementById("whatif-bmi").value),
      HighBP: document.getElementById("whatif-HighBP").checked ? 1 : 0,
      HighChol: document.getElementById("whatif-HighChol").checked ? 1 : 0,
      Smoker: document.getElementById("whatif-Smoker").checked ? 1 : 0,
      PhysActivity: document.getElementById("whatif-PhysActivity").checked ? 1 : 0,
      Fruits: document.getElementById("whatif-Fruits").checked ? 1 : 0,
    };
```

bằng

```javascript
    const whatIfChanges = {
      BMI: parseFloat(document.getElementById("whatif-bmi").value),
      Smoker: document.getElementById("whatif-Smoker").checked ? 1 : 0,
      PhysActivity: document.getElementById("whatif-PhysActivity").checked ? 1 : 0,
      Fruits: document.getElementById("whatif-Fruits").checked ? 1 : 0,
      Veggies: document.getElementById("whatif-Veggies").checked ? 1 : 0,
      HvyAlcoholConsump: document.getElementById("whatif-HvyAlcoholConsump").checked ? 1 : 0,
    };
```

Lưu ý: giữ nguyên định dạng `const whatIfChanges = {` … `};` vì test hợp đồng ở F01.3 đọc khối này bằng regex.

(c) Trong `callExplain`, thay

```javascript
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      return;
    }
```

bằng

```javascript
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      if (whatIfChanges !== null) {
        document.getElementById("whatif-note").textContent =
          "Không cập nhật được kịch bản what-if (mã lỗi " + res.status + "). Vui lòng thử lại.";
      }
      return;
    }
```

**Bước F01.3 — test hợp đồng giao diện ↔ máy chủ.** Mở `tests/test_demo_static.py`, thêm vào cuối (import `re` và `load_config` ở đầu file, cạnh các import hiện có):

```python
import re

from diabetes_ai.config import load_config


def test_whatif_keys_sent_by_ui_are_allowed_by_server() -> None:
    """Các khóa what-if mà app.js gửi tới /explain phải nằm trong khối `what_if` của configs/features.yaml."""
    js = (ROOT / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    allowed = set(load_config("features")["what_if"].keys())

    block = re.search(r"const whatIfChanges = \{(.*?)\n\s*\};", js, re.S)
    assert block, "Không tìm thấy khối `const whatIfChanges = {...}` trong app.js"
    sent = set(re.findall(r"^\s*(\w+):", block.group(1), re.M))
    assert sent, "Khối whatIfChanges không có khóa nào"
    assert sent <= allowed, f"app.js gửi khóa không được phép: {sorted(sent - allowed)}"

    for key in sent - {"BMI"}:
        assert f'id="whatif-{key}"' in html, f"Thiếu ô bật/tắt whatif-{key} trong index.html"
```

**Kiểm chứng F01:**

```bash
python -m pytest -q tests/test_demo_static.py            # kỳ vọng: 4 passed (3 cũ + 1 mới)
```

Nếu muốn thấy test mới bắt được lỗi cũ: `git stash` hai file `app/*`, chạy lại → test mới phải **fail** với thông báo `app.js gửi khóa không được phép: ['HighBP', 'HighChol']`; rồi `git stash pop`.

Kiểm tra bằng tay: `uvicorn diabetes_ai.serving.app:app --port 8200`, mở `http://localhost:8200/app/`, bấm **Đánh giá nguy cơ**, kéo thanh BMI và bật/tắt từng ô: bảng "Trước / Sau / Chênh lệch" phải đổi giá trị; tab Network của trình duyệt chỉ có `/explain` mã 200.

**Commit:** `fix(demo): what-if chỉ gửi các khóa lối sống được phép và hiển thị lỗi`

---

### F02 — `/explain` trả 500 với đầu vào sai; `NaN` lọt qua kiểm tra miền

**Bằng chứng:**
- `serving/routes.py::explain` gọi `predictor.explain(...)` không có `try/except`; `what_if_custom` ném `ValueError` khi khóa lạ hoặc giá trị ngoài `FEATURE_RANGES`. `serving/app.py` chỉ có handler cho `RequestValidationError`. Kết quả: 500, không phải 422.
- Kiểm tra miền `if val < min_val or val > max_val` **không bắt được `NaN`** (mọi so sánh với `NaN` là `False`) → `x_new[...] = NaN` → xác suất `NaN` → lỗi tuần tự hóa JSON → 500. `ExplainRequest.what_if_changes: dict[str, float] | None` không ràng buộc gì (JSON `NaN` được Python `json.loads` chấp nhận).
- Handler 422 hiện có đưa nguyên `input` vào `JSONResponse`; nếu `input` chứa `NaN` thì chính handler cũng sinh lỗi 500 (`allow_nan=False`).

**Bước F02.1 — `src/diabetes_ai/explain/what_if.py`.**
(a) Ngay **trước** dòng `def what_if_custom(`, thêm lớp lỗi riêng (kế thừa `ValueError` nên test cũ dùng `pytest.raises(ValueError)` vẫn đúng):

```python
class WhatIfError(ValueError):
    """Lỗi đầu vào của kịch bản what-if tùy biến (khóa không được phép hoặc giá trị không hợp lệ)."""


```

(b) Trong thân `what_if_custom`, thay `raise ValueError(` thứ nhất:

```python
        raise ValueError(
            f"Chỉ được thay đổi: {allowed_str}. Đặc trưng không được phép: {sorted(invalid_feats)}"
        )
```

bằng

```python
        raise WhatIfError(
            f"Chỉ được thay đổi: {allowed_str}. Đặc trưng không được phép: {sorted(invalid_feats)}"
        )
```

(c) Thay vòng kiểm tra miền:

```python
    for feat, val in changes.items():
        if feat in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[feat]
            if val < min_val or val > max_val:
                raise ValueError(
                    f"Giá trị của {feat} ({val}) nằm ngoài khoảng cho phép [{min_val}, {max_val}]."
                )
```

bằng

```python
    for feat, val in changes.items():
        if not np.isfinite(val):
            raise WhatIfError(f"Giá trị của {feat} phải là số hữu hạn (nhận được {val}).")
        if feat in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[feat]
            if val < min_val or val > max_val:
                raise WhatIfError(
                    f"Giá trị của {feat} ({val}) nằm ngoài khoảng cho phép [{min_val}, {max_val}]."
                )
```

Cập nhật dòng `Raises:` trong docstring của `what_if_custom` thành `WhatIfError: Nếu có đặc trưng nằm ngoài danh sách allowed, giá trị không hữu hạn hoặc nằm ngoài FEATURE_RANGES.` Kiểm tra `np` đã được import ở đầu file (`grep -n "^import numpy" src/diabetes_ai/explain/what_if.py`).

**Bước F02.2 — `src/diabetes_ai/serving/routes.py`.** Thay dòng import FastAPI và thêm import `WhatIfError`:

```python
from fastapi import APIRouter, Depends, HTTPException

from diabetes_ai.explain.what_if import WhatIfError
```

(đặt dòng thứ hai cùng nhóm import `diabetes_ai.*`). Thay hàm `explain`:

```python
@router.post("/explain", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
    predictor: Predictor = Depends(get_predictor),
) -> ExplainResponse:
    """Giải thích các yếu tố rủi ro và mô phỏng kịch bản what-if."""
    try:
        result = predictor.explain(request.profile, what_if_changes=request.what_if_changes)
    except WhatIfError as exc:
        # Đầu vào what-if sai (khóa không được phép / ngoài miền / không hữu hạn) là lỗi của client.
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "field": "what_if_changes",
                    "message": str(exc),
                    "input": request.what_if_changes,
                }
            ],
        ) from exc
    return ExplainResponse(**result)
```

Dạng `{"detail": [{"field", "message", "input"}]}` giống hệt handler 422 hiện có nên giao diện xử lý như nhau.

**Bước F02.3 — `src/diabetes_ai/inference/schemas.py`.** Thêm `import math` (ngay sau `from __future__ import annotations`, trước `from typing import Any, Literal`), đổi dòng import pydantic thành

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator
```

và thêm vào cuối lớp `ExplainRequest` (sau trường `what_if_changes`):

```python
    @field_validator("what_if_changes")
    @classmethod
    def _check_finite(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        """Từ chối NaN/Infinity trong các giá trị what-if ngay ở lớp lược đồ."""
        if value is None:
            return value
        for key, val in value.items():
            if not math.isfinite(val):
                raise ValueError(f"Giá trị của {key} phải là số hữu hạn.")
        return value
```

**Bước F02.4 — `src/diabetes_ai/serving/app.py`.** Handler 422 phải không sinh lỗi khi `input` chứa `NaN`/đối tượng lạ.
(a) Đảm bảo đầu file có `import math` và `from typing import Any` (thêm nếu chưa có).
(b) Thêm hàm ngay trước `@app.exception_handler(RequestValidationError)`:

```python
def _json_safe(value: Any) -> Any:
    """Chuyển giá trị không tuần tự hóa được (NaN/Infinity, đối tượng lạ) thành chuỗi."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


```

(c) Trong handler, thay `"input": input_val,` bằng `"input": _json_safe(input_val),`.

**Bước F02.5 — test.** Thêm vào cuối `tests/test_api_predict.py` (đã có `pytest`, `TestClient`, fixture `client`, `sample_profile`, và fixture tự động nạp `tiny_bundle`; thêm `import json` ở đầu file):

```python
def test_explain_rejects_disallowed_what_if_feature_422(client: TestClient, sample_profile: dict) -> None:
    """Khóa không nằm trong khối what_if (HighBP) phải bị từ chối bằng 422, không phải 500."""
    res = client.post("/explain", json={"profile": sample_profile, "what_if_changes": {"HighBP": 0}})
    assert res.status_code == 422
    assert "HighBP" in res.text


def test_explain_rejects_out_of_range_what_if_422(client: TestClient, sample_profile: dict) -> None:
    """BMI = 500 nằm ngoài miền → 422."""
    res = client.post("/explain", json={"profile": sample_profile, "what_if_changes": {"BMI": 500}})
    assert res.status_code == 422
    assert "BMI" in res.text


def test_explain_rejects_nan_what_if_422(client: TestClient, sample_profile: dict) -> None:
    """JSON chứa NaN phải bị từ chối bằng 422 và máy chủ không sập khi tuần tự hóa lỗi."""
    body = json.dumps({"profile": sample_profile, "what_if_changes": {"BMI": float("nan")}})
    res = client.post("/explain", content=body, headers={"Content-Type": "application/json"})
    assert res.status_code == 422


def test_explain_accepts_every_key_the_ui_sends(client: TestClient, sample_profile: dict) -> None:
    """Sáu khóa lối sống hợp lệ (đúng bộ giao diện gửi) phải trả 200 và có custom_what_if."""
    changes = {"BMI": 24.5, "Smoker": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0}
    res = client.post("/explain", json={"profile": sample_profile, "what_if_changes": changes})
    assert res.status_code == 200
    assert res.json()["custom_what_if"] is not None
```

**Kiểm chứng F02:**

```bash
python -m pytest -q tests/test_api_predict.py tests/test_predictor.py tests/test_what_if.py tests/test_demo_static.py
# kỳ vọng: toàn bộ passed; test_predictor.py::...Age... vẫn pass (WhatIfError là con của ValueError)
```

**Commit:** `fix(api): /explain trả 422 cho what-if không hợp lệ và chặn NaN`

---

### F03 — `SafetyAgent` đọc sai khóa xác suất tiểu đường

**Bằng chứng:** `src/diabetes_ai/agents/safety_agent.py` dùng `if "Tiểu đường" in probs` (chữ T hoa) rồi `elif "1" in probs`. Tên lớp thật là `CLASSES = ["không", "tiền tiểu đường", "tiểu đường"]` (`data/schema.py`; xác nhận qua bundle: `class_names = ['không', 'tiền tiểu đường', 'tiểu đường']`), và `Predictor.predict` trả `probabilities = {tên_lớp: p}`. Hai nhánh đều không bao giờ khớp → `p_diabetes` rơi về `risk_index` (= P(tiểu đường) + 0,5·P(tiền tiểu đường)). Hệ quả: luật **R10** (`p_diabetes >= 0.5`) trong luồng đa agent kích hoạt theo chỉ số nguy cơ chứ không theo xác suất tiểu đường, khác với `Predictor.predict` (dùng `proba[2]`).

**Bước F03.1 — `src/diabetes_ai/agents/safety_agent.py`.**
(a) Thêm import (cạnh các import `diabetes_ai.*` hiện có):

```python
from diabetes_ai.data.schema import CLASSES
```

(b) Thêm phương thức tĩnh vào lớp `SafetyAgent`, ngay **trước** dòng `def run(` (định vị bằng `grep -n "def run" src/diabetes_ai/agents/safety_agent.py`):

```python
    @staticmethod
    def _extract_p_diabetes(probs: dict[str, Any], fallback: float) -> float:
        """Lấy P(tiểu đường) từ từ điển xác suất theo tên lớp, đồng nhất với Predictor.predict.

        - Đa lớp: lớp cuối của CLASSES ("tiểu đường"), so khớp không phân biệt hoa/thường.
        - Nhị phân (đúng 2 giá trị): cột dương (cột thứ hai).
        - Không xác định được: dùng `fallback` (chỉ số nguy cơ).
        """
        wanted = CLASSES[-1].casefold()
        for name, value in probs.items():
            if str(name).casefold() == wanted:
                return float(value)
        if len(probs) == 2:
            return float(list(probs.values())[1])
        return float(fallback)

```

(c) Trong `run`, thay

```python
        # Xác định xác suất tiểu đường và chỉ số nguy cơ cho ngữ cảnh đánh giá luật
        probs = predictor_output.get("probabilities", {})
        if "Tiểu đường" in probs:
            p_diabetes = float(probs["Tiểu đường"])
        elif "1" in probs:
            p_diabetes = float(probs["1"])
        else:
            p_diabetes = float(risk_result.get("risk_index", 0.0))
        r_idx = float(predictor_output.get("risk_index", 0.0))
```

bằng

```python
        # Xác định xác suất tiểu đường và chỉ số nguy cơ cho ngữ cảnh đánh giá luật
        probs = predictor_output.get("probabilities", {}) or {}
        r_idx = float(predictor_output.get("risk_index", 0.0))
        p_diabetes = self._extract_p_diabetes(probs, fallback=r_idx)
```

**Bước F03.2 — test.** Thêm vào cuối `tests/test_agents.py` (thêm `from diabetes_ai.agents.safety_agent import SafetyAgent` cạnh import `Orchestrator`):

```python
def _safety_state(probs: dict[str, float], risk_index: float) -> dict:
    return {
        "profile": _load_sample_profile(),
        "risk_result": {"risk_level": "cao", "risk_index": risk_index, "probabilities": probs},
    }


def test_safety_agent_r10_uses_true_diabetes_probability():
    """R10 (p_diabetes >= 0.5) phải dựa trên P(tiểu đường), không phải chỉ số nguy cơ."""
    agent = SafetyAgent()
    # risk_index = 0.62 nhưng P(tiểu đường) chỉ 0.30 → R10 KHÔNG được kích hoạt
    low = agent.run(_safety_state({"không": 0.30, "tiền tiểu đường": 0.40, "tiểu đường": 0.30}, 0.62))
    assert "R10" not in [a.rule_id for a in low["alerts"]]
    # P(tiểu đường) = 0.55 → R10 phải kích hoạt
    high = agent.run(_safety_state({"không": 0.25, "tiền tiểu đường": 0.20, "tiểu đường": 0.55}, 0.65))
    assert "R10" in [a.rule_id for a in high["alerts"]]
```

Nếu `cross_check` (critic) đòi thêm khóa trong `predictor_output`, bổ sung vào `_safety_state` đúng khóa mà lỗi báo — không sửa `critic.py`.

**Kiểm chứng F03:**

```bash
python -m pytest -q tests/test_agents.py tests/test_critic.py tests/test_safety_rules.py
# kỳ vọng: passed (test_agents_do_not_call_each_other vẫn pass: import data.schema không chứa tên agent khác)
```

**Commit:** `fix(agents): SafetyAgent lấy P(tiểu đường) theo tên lớp thật`

---

### F04 — Test phụ thuộc mô hình thật (không có trong git) → CI/bản clone sạch bị fail

**Bằng chứng:** `tests/test_agents.py` gọi `Orchestrator()` (3 chỗ, dòng 20, 59, 74). Mặc định `RiskAgent` gọi `Predictor.from_bundle()` → đọc `models/best_model.joblib`. `.gitignore` bỏ qua `*.joblib` và `models/*` nên file này không có trên bản clone sạch (CI) → `ModelNotReady`. Các test khác trong `test_api_predict.py` đã đúng cách: dùng fixture `tiny_bundle` + `MODEL_PATH`.

**Bước F04.1 — `tests/test_agents.py`.** Thêm (cạnh import hiện có `import ast`, `import json`; thêm `import pytest`) và đặt ngay sau các import:

```python
@pytest.fixture(autouse=True)
def _use_tiny_bundle(monkeypatch: pytest.MonkeyPatch, tiny_bundle) -> None:
    """Ép Orchestrator/RiskAgent dùng bundle nhỏ của fixture, không phụ thuộc models/best_model.joblib."""
    monkeypatch.setenv("MODEL_PATH", str(tiny_bundle))
```

(`tiny_bundle` là fixture trong `tests/conftest.py`; `Predictor.from_bundle()` đọc `MODEL_PATH` khi không truyền đường dẫn.)

**Bước F04.2 — kiểm tra toàn bộ bộ test trên bản clone sạch** (mô phỏng CI; dùng chính venv hiện tại vì `pyproject.toml` đặt `pythonpath = ["src"]` nên mã trong bản clone được ưu tiên):

```bash
rm -rf /tmp/dal-clean && git clone . /tmp/dal-clean
cd /tmp/dal-clean && python -m pytest -q 2>&1 | tail -25
cd - >/dev/null
```

Kỳ vọng: không test nào fail (các test `real_data` được `skip`). Nếu còn test fail do thiếu artifact (mô hình, `indices.npz`, CSV, `reports/metrics/*`), sửa **từng test** cho tự đủ bằng `tmp_path`/`tiny_bundle`/`synthetic_df` (xem `tests/conftest.py`); **không** dùng `skip` để che lỗi và không commit artifact vào git. Ghi danh sách test đã sửa vào thông điệp commit.

**Kiểm chứng F04:**

```bash
mv models/best_model.joblib /tmp/best_model.joblib.bak
python -m pytest -q tests/test_agents.py tests/test_api_predict.py      # kỳ vọng: passed dù không có mô hình thật
mv /tmp/best_model.joblib.bak models/best_model.joblib
```

**Commit:** `test(agents): dùng bundle nhỏ để bộ test chạy được trên bản clone sạch`

---

## 4. Nhóm P1 — phần còn thiếu so với kế hoạch và hạ tầng

### F05 — Hoàn tất K02c: notebook EDA, hình `eda_*.png`, từ điển dữ liệu

**Bằng chứng:** `notebooks/01_eda_brfss.ipynb` chỉ 195 byte (stub); `reports/figures/` không có `eda_*.png`; `docs/data_dictionary.md` chỉ 129 byte; `docs/roadmap.md` 127 byte. Tiêu chí K02 trong `plan.md`: notebook chạy hết, ≥ 14 ô, ≥ 6 hình `eda_*.png`, `grep -c "Traceback"` = 0, `grep -c "include_test"` = 0, từ điển chứa 253.680, 23.899, 172.502, 30.442, 50.736. **Mọi thống kê liên quan nhãn chỉ tính trên tập train**; phần chất lượng dữ liệu dùng toàn bộ.

**Bước F05.1 — tạo notebook bằng script tạm** (không commit script; `nbformat` đã có cùng `jupyterlab`). Lưu đoạn sau vào `/tmp/make_eda_nb.py` rồi chạy `python /tmp/make_eda_nb.py` từ thư mục gốc dự án:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("# EDA — BRFSS 2015 (Diabetes Health Indicators)\n\n"
   "> Chỉ để học tập, không phải chẩn đoán y khoa.\n\n"
   "Quy tắc: mọi thống kê liên quan nhãn (tỉ lệ bệnh theo nhóm, tương quan) chỉ tính trên **tập train**; "
   "phần chất lượng dữ liệu dùng toàn bộ bảng. Notebook này không nạp tập test.")

code('''import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diabetes_ai.data.load import load_raw
from diabetes_ai.data.schema import CLASSES, FEATURES, TARGET
from diabetes_ai.data.split import load_split
from diabetes_ai.features.feature_names import (
    AGE_LABELS, EDUCATION_LABELS, GENHLTH_LABELS, INCOME_LABELS, VI_FEATURE,
)
from diabetes_ai.paths import FIGURES_DIR, METRICS_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
df = load_raw()
split = load_split("full")
train = df.iloc[split.train].reset_index(drop=True)   # thống kê liên quan nhãn chỉ dùng train
print(len(df), len(train), split.sizes())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=100)
    plt.close(fig)


def rate_diabetes(col):
    """Tỉ lệ nhãn 2 (tiểu đường) theo giá trị của cột, chỉ trên train."""
    return (train[TARGET] == 2).groupby(train[col].astype(int)).mean()
''')

md("## 1. Chất lượng dữ liệu (toàn bộ bảng)")
code('''val_json = json.loads((METRICS_DIR / "data_validation.json").read_text(encoding="utf-8"))
quality = {
    "so_dong": len(df),
    "o_thieu": int(df.isna().sum().sum()),
    "dong_trung_hoan_toan": int(df.duplicated().sum()),
    "BMI_lon_hon_60": int((df["BMI"] > 60).sum()),
}
print(quality)
assert quality["dong_trung_hoan_toan"] == val_json["n_duplicates"]
''')

md("## 2. Phân phối nhãn (train)")
code('''counts = train[TARGET].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(CLASSES, counts.values)
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom", fontsize=8)
ax.set_title("Phân phối nhãn 3 lớp (train)")
ax.set_ylabel("Số hồ sơ")
save(fig, "eda_01_label_distribution.png")
print((counts / counts.sum()).round(4).to_dict())
''')

md("## 3. BMI theo nhóm nhãn (train)")
code('''fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([train.loc[train[TARGET] == k, "BMI"] for k in range(3)], labels=CLASSES, showfliers=False)
ax.set_title("BMI theo nhóm nhãn (train, ẩn điểm ngoại lai)")
ax.set_ylabel("BMI")
save(fig, "eda_02_bmi_by_class.png")
print(train.groupby(TARGET)["BMI"].median().round(1).to_dict())
''')

md("## 4. Tỉ lệ tiểu đường theo nhóm tuổi (train)")
code('''r = rate_diabetes("Age")
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar([AGE_LABELS[k] for k in r.index], r.values * 100)
ax.set_title("Tỉ lệ tiểu đường (%) theo nhóm tuổi (train)")
ax.set_ylabel("%")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
save(fig, "eda_03_prevalence_by_age.png")
''')

md("## 5. Tỉ lệ tiểu đường theo thu nhập và học vấn (train)")
code('''fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, col, labels, title in [
    (axes[0], "Income", INCOME_LABELS, "Thu nhập"),
    (axes[1], "Education", EDUCATION_LABELS, "Học vấn"),
]:
    r = rate_diabetes(col)
    ax.bar([str(labels[k]) for k in r.index], r.values * 100)
    ax.set_title(f"Tỉ lệ tiểu đường (%) theo {title}")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
save(fig, "eda_04_prevalence_by_income_education.png")
''')

md("## 6. Tỉ lệ tiểu đường theo tự đánh giá sức khỏe (train)")
code('''r = rate_diabetes("GenHlth")
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.bar([str(GENHLTH_LABELS[k]) for k in r.index], r.values * 100)
ax.set_title("Tỉ lệ tiểu đường (%) theo GenHlth (train)")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
save(fig, "eda_05_prevalence_by_genhlth.png")
''')

md("## 7. Yếu tố nhị phân: chênh lệch tỉ lệ tiểu đường khi có / không có yếu tố (train)")
code('''bin_cols = [c for c in FEATURES if train[c].nunique() == 2]
is_dia = train[TARGET] == 2
gap = pd.Series({c: 100 * (is_dia[train[c] == 1].mean() - is_dia[train[c] == 0].mean()) for c in bin_cols}).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh([VI_FEATURE[c] for c in gap.index], gap.values)
ax.set_title("Chênh lệch tỉ lệ tiểu đường (điểm %): có − không (train)")
save(fig, "eda_06_binary_factor_gap.png")
''')

md("## 8. Ma trận tương quan (train)")
code('''corr = train[FEATURES + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr)))
ax.set_yticklabels(corr.columns, fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Tương quan Pearson (train)")
save(fig, "eda_07_correlation.png")
''')

md("## 9. Kết luận (số liệu in từ kết quả chạy)")
code('''top = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(5).index
print("Tỉ lệ tiểu đường nền (train): %.2f%%" % (100 * (train[TARGET] == 2).mean()))
print("5 đặc trưng tương quan mạnh nhất với nhãn:", {k: round(float(corr.loc[k, TARGET]), 3) for k in top})
print("Lưu ý: tương quan không phải nhân quả; dữ liệu khảo sát tự khai ở Hoa Kỳ năm 2015.")
''')

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/01_eda_brfss.ipynb")
print("Số ô:", len(cells))
```

Số ô tạo ra: 18 (≥ 14). Số hình: 7 (≥ 6). Nếu `load_raw()` / `load_split()` có chữ ký khác (đã kiểm: `load_raw(path=None, config=None)`, `load_split(variant="full", seed=42, splits_dir=None)` → `SplitIndices` có `.train/.val/.test`), sửa cho khớp; **không** truy cập `split.test` trong notebook.

**Bước F05.2 — chạy notebook và kiểm tra.**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_brfss.ipynb
python -c "import glob,json; n=len(glob.glob('reports/figures/eda_*.png')); c=len(json.load(open('notebooks/01_eda_brfss.ipynb',encoding='utf-8'))['cells']); print(n,c)"
grep -c "Traceback" notebooks/01_eda_brfss.ipynb     # kỳ vọng 0
grep -c "include_test" notebooks/01_eda_brfss.ipynb  # kỳ vọng 0
```

Kỳ vọng: lệnh đầu thoát 0; in `7 18`; hai `grep` in `0`. Nếu thiếu kernel: `python -m ipykernel install --user --name python3` rồi chạy lại. Nếu `boxplot(..., labels=...)` báo cảnh báo đổi tên tham số ở phiên bản matplotlib mới, đổi `labels=` thành `tick_labels=` (giữ nguyên ý nghĩa).

**Bước F05.3 — sinh từ điển dữ liệu từ số liệu đo** (không gõ tay số). Tạo `scripts/make_data_dictionary.py` (file mới, docstring tiếng Việt):

```python
"""Sinh docs/data_dictionary.md từ số liệu đã đo (không gõ tay số liệu)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from diabetes_ai.data.schema import CLASSES, COUNT_FEATURES, FEATURE_RANGES, FEATURES
from diabetes_ai.features.feature_names import (
    AGE_LABELS,
    EDUCATION_LABELS,
    GENHLTH_LABELS,
    INCOME_LABELS,
    SEX_LABELS,
    VI_FEATURE,
)
from diabetes_ai.paths import DOCS_DIR, METRICS_DIR, RAW_DIR, SPLITS_DIR

CODE_TABLES = {
    "Age": AGE_LABELS,
    "Education": EDUCATION_LABELS,
    "GenHlth": GENHLTH_LABELS,
    "Income": INCOME_LABELS,
    "Sex": SEX_LABELS,
}


def vn(n: int | float) -> str:
    """Định dạng số nguyên theo kiểu Việt Nam (dấu chấm ngăn nghìn)."""
    return f"{int(n):,}".replace(",", ".")


def pct(x: float) -> str:
    """Định dạng phần trăm với dấu phẩy thập phân."""
    return f"{100 * x:.1f}".replace(".", ",") + "%"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _kind(name: str, lo: float, hi: float) -> str:
    if (lo, hi) == (0.0, 1.0):
        return "nhị phân"
    if name in COUNT_FEATURES:
        return "đếm (số ngày)"
    if name == "BMI":
        return "liên tục"
    return "thứ bậc"


def _coding(name: str, lo: float, hi: float) -> str:
    table = CODE_TABLES.get(name)
    if table:
        return "; ".join(f"{k} = {v}" for k, v in table.items())
    return "0 = không, 1 = có" if (lo, hi) == (0.0, 1.0) else f"{lo:g} – {hi:g}"


def main() -> None:
    val = _read(METRICS_DIR / "data_validation.json")
    csv_info = _read(METRICS_DIR / "env_check.json")["csv"]
    full = _read(SPLITS_DIR / "full_seed42" / "manifest.json")
    dedup = _read(SPLITS_DIR / "dedup_seed42" / "manifest.json")
    csv_path = next(RAW_DIR.glob("*.csv"))
    downloaded = datetime.fromtimestamp(csv_path.stat().st_mtime).strftime("%Y-%m-%d")

    L: list[str] = []
    L += ["# Từ điển dữ liệu — BRFSS 2015 (Diabetes Health Indicators)", ""]
    L += ["> Tệp này do `scripts/make_data_dictionary.py` sinh từ số liệu đo; không sửa tay các con số.", ""]
    L += ["## 1. Nguồn", ""]
    L += [f"- Tệp: `{csv_path.name}` (bộ \"Diabetes Health Indicators\" từ khảo sát BRFSS 2015 của CDC, bản đăng trên Kaggle).",
          f"- Ngày tệp (theo `stat`): {downloaded}; kích thước {vn(csv_info['bytes'])} byte; sha256 `{csv_info['sha256']}`.",
          f"- Số dòng: {vn(val['n_rows'])}; số cột: {val['n_cols']} (21 đặc trưng + 1 nhãn).", ""]
    L += ["## 2. Các cột", ""]
    L += ["| Cột | Tiếng Việt | Kiểu | Miền | Mã hóa |", "|---|---|---|---|---|"]
    for name in FEATURES:
        lo, hi = FEATURE_RANGES[name]
        L.append(f"| `{name}` | {VI_FEATURE[name]} | {_kind(name, lo, hi)} | {lo:g} – {hi:g} | {_coding(name, lo, hi)} |")
    L += ["", "**Lưu ý (giả định A15):** cách mã hóa Age/Income/Education/GenHlth/Sex theo mô tả của bộ dữ liệu Kaggle; "
          "**chưa đối chiếu** với codebook CDC gốc.", ""]
    L += ["## 3. Nhãn", ""]
    L += [f"- Nhãn 3 lớp `Diabetes_012`: " + "; ".join(f"{i} = {c}" for i, c in enumerate(CLASSES)) + ".",
          "- Nhãn nhị phân: 0 = không; 1 = tiền tiểu đường hoặc tiểu đường.",
          "- Số lượng toàn bộ: " + "; ".join(f"{CLASSES[int(k)]} = {vn(v)}" for k, v in val["label_counts"].items()) + ".", ""]
    L += ["## 4. Chất lượng dữ liệu", ""]
    L += [f"- Dòng trùng hoàn toàn (cả nhãn): {vn(val['n_duplicates'])} ({pct(val['n_duplicates'] / val['n_rows'])}); "
          f"còn {vn(val['n_unique_rows'])} dòng duy nhất.",
          f"- Kết luận kiểm tra tự động: `ok = {val['ok']}`.", ""]
    L += ["## 5. Quyết định về dòng trùng lặp", ""]
    L += [f"- Biến thể **`full`** (giữ dòng trùng) là biến thể **chính**: các dòng trùng là những người trả lời khác nhau "
          f"có hồ sơ giống nhau; khử trùng làm lệch tỉ lệ tiểu đường nền từ {pct(full['class_pct']['train']['2'])} "
          f"sang {pct(dedup['class_pct']['train']['2'])} và làm xác suất hiệu chuẩn sai.",
          "- Biến thể **`dedup`** (xóa bản sao thừa, giữ lần xuất hiện đầu tiên) chỉ là **đối chứng**.", ""]
    L += ["## 6. Chia tập (seed 42, phân tầng theo nhãn 3 lớp)", ""]
    L += ["| Biến thể | Train | Validation | Test |", "|---|---|---|---|"]
    for name, m in (("full", full), ("dedup", dedup)):
        s = m["sizes"]
        L.append(f"| {name} | {vn(s['train'])} | {vn(s['val'])} | {vn(s['test'])} |")
    L += ["", "`val_size = 0,15` là 15% của phần còn lại sau khi tách test (`test_size = 0,2`)."]

    out = DOCS_DIR / "data_dictionary.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
```

Chạy: `python scripts/make_data_dictionary.py`. Nếu `val['label_counts']` hoặc `val['ok']` có cấu trúc khác khi chạy, sửa script cho khớp tệp thật (đã xác minh: `label_counts = {"0": 213703, "1": 4631, "2": 35346}`, `n_duplicates = 23899`, `n_unique_rows = 229781`, `ok = true`).

**Bước F05.4 — test.** Tạo `tests/test_data_dictionary.py`:

```python
"""Kiểm tra từ điển dữ liệu chứa đúng các con số đã đo."""

import json

from diabetes_ai.paths import DOCS_DIR, METRICS_DIR, SPLITS_DIR


def _vn(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def test_data_dictionary_has_measured_numbers() -> None:
    text = (DOCS_DIR / "data_dictionary.md").read_text(encoding="utf-8")
    val = json.loads((METRICS_DIR / "data_validation.json").read_text(encoding="utf-8"))
    sizes = json.loads((SPLITS_DIR / "full_seed42" / "manifest.json").read_text(encoding="utf-8"))["sizes"]
    for n in (val["n_rows"], val["n_duplicates"], sizes["train"], sizes["val"], sizes["test"]):
        assert _vn(n) in text, f"Thiếu số {_vn(n)} trong docs/data_dictionary.md"
```

**Bước F05.5 — `docs/roadmap.md`.** Thay nội dung stub bằng bản ngắn (tiếng Việt): trạng thái K01–K10 đã xong, K11 tùy chọn đã có mã; việc còn lại = các mục trong `fixing.md`. Không thêm số liệu mới.

**Kiểm chứng F05:**

```bash
grep -c "23.899" docs/data_dictionary.md              # >= 1
for n in 253.680 172.502 30.442 50.736; do grep -c "$n" docs/data_dictionary.md; done   # mỗi số >= 1
python -m pytest -q tests/test_data_dictionary.py     # 1 passed
git status --short                                    # chỉ gồm notebook, 7 PNG, data_dictionary.md, roadmap.md, script và test mới
```

**Commit:** hai commit — `docs(eda): notebook EDA và 7 hình eda_*.png` rồi `docs(data): từ điển dữ liệu sinh tự động và roadmap`.

---

### F06 — `.env` không bao giờ được nạp; `.env.example` có biến thừa

**Bằng chứng:** `config.get_env()` (nạp `.env` bằng `python-dotenv`) **không có nơi nào gọi** (`grep -rn "get_env(" src scripts` chỉ ra dòng định nghĩa). `serving/deps.py` và `inference/predictor.py` đọc `os.environ.get("MODEL_PATH")` trực tiếp, nên `MODEL_PATH` đặt trong `.env` (đúng như `.env.example` hướng dẫn) bị bỏ qua. `APP_ENV`, `API_HOST`, `API_PORT`, `RANDOM_SEED` trong `.env.example` không được mã nào đọc.

**Bước F06.1 — `src/diabetes_ai/serving/deps.py`.**
- Đổi `from diabetes_ai.config import load_config` thành `from diabetes_ai.config import get_env, load_config`.
- Thay `model_path = os.environ.get("MODEL_PATH")` bằng `model_path = get_env("MODEL_PATH")`.
- Nếu sau đó `os` không còn được dùng (`grep -n "os\." src/diabetes_ai/serving/deps.py` rỗng) thì xóa dòng `import os` (tránh lỗi ruff F401).

**Bước F06.2 — `src/diabetes_ai/inference/predictor.py`.**
- Đổi `from diabetes_ai.config import load_config` thành `from diabetes_ai.config import get_env, load_config`.
- Thay `env_path = os.environ.get("MODEL_PATH")` bằng `env_path = get_env("MODEL_PATH")`.
- Xóa `import os` nếu không còn dùng (`grep -n "os\." src/diabetes_ai/inference/predictor.py`).

**Bước F06.3 — cô lập test khỏi `.env` của lập trình viên.** `get_env` ghi `.env` vào `os.environ` một lần cho cả tiến trình. Trong `tests/conftest.py` thêm fixture tự động:

```python
@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Không để file .env cục bộ của lập trình viên ảnh hưởng tới kết quả test."""
    monkeypatch.setattr("diabetes_ai.config._DOTENV_LOADED", True)
```

**Bước F06.4 — `.env.example`.** Xác nhận không có nơi nào đọc 4 biến thừa (`grep -rn "APP_ENV\|API_HOST\|API_PORT\|RANDOM_SEED" src scripts configs` rỗng), rồi thay toàn bộ nội dung bằng:

```
# Sao chép thành .env, KHÔNG commit .env
# Đường dẫn gói mô hình phục vụ (mặc định lấy từ configs/serve.yaml)
MODEL_PATH=models/best_model.joblib
# (Tùy chọn) thư mục gốc dự án nếu chạy mã từ nơi khác
# DIABETES_AI_ROOT=
# Cổng và host chạy qua tham số của uvicorn: --host 127.0.0.1 --port 8200
```

**Bước F06.5 — test.** Tạo `tests/test_env_config.py`:

```python
"""Kiểm tra get_env nạp được biến từ file .env."""

import os

import diabetes_ai.config as cfg


def test_get_env_reads_dotenv(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("MODEL_PATH=models/from_dotenv.joblib\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    monkeypatch.setattr(cfg, "_DOTENV_LOADED", False)
    os.environ.pop("MODEL_PATH", None)
    try:
        assert cfg.get_env("MODEL_PATH") == "models/from_dotenv.joblib"
    finally:
        os.environ.pop("MODEL_PATH", None)  # load_dotenv ghi vào os.environ, phải dọn
```

**Kiểm chứng F06:**

```bash
python -m pytest -q tests/test_env_config.py tests/test_api_predict.py tests/test_predictor.py
python -m ruff check src tests          # không phát sinh lỗi F401 mới
```

**Commit:** `fix(config): nạp MODEL_PATH qua get_env để .env có hiệu lực`

---

### F07 — Tài liệu lệch với code và số đo

**Bằng chứng (đối chiếu số liệu đo trong `reports/metrics/*.json`):**

| Tài liệu | Đang ghi | Thực tế |
|---|---|---|
| `README.md` dòng 41 | chia "stratified 60/20/20" | 68% / 12% / 20% (172.502 / 30.442 / 50.736); `val_size=0,15` là 15% của phần còn lại sau khi tách test |
| `README.md` dòng 104 | loại 24.206 bản ghi trùng | 23.899 dòng trùng (9,42%), còn 229.781 |
| `docs/architecture.md` dòng 72 | 24.206 dòng trùng, còn 229.474 | 23.899; còn 229.781 |
| `docs/architecture.md` dòng 33 | hàm `stratified_split` | hàm thật là `make_split` (`data/split.py`) |
| `docs/architecture.md` dòng 35 | MLP "Dropout/BatchNorm" | MLP 21→128→64→32→3 chỉ có ReLU + Dropout (xác nhận: `grep -n "BatchNorm" src/diabetes_ai/models/mlp.py` rỗng) |
| `docs/architecture.md` dòng 40 | endpoint `/explain/global` | API có `/health`, `/model-info`, `/predict`, `/explain` (không có `/explain/global`) |
| `reports/final_report.md` §6 (dòng ~121) | permutation importance "trên tập dữ liệu kiểm thử" | tính trên **validation** (`scripts/explain_global.py`: `X=data.X_val`, `include_test=False`) |

**Bước F07.1 — sửa bằng hàm `sub()` (Phụ lục A; giữ nguyên CRLF).** Chạy từ thư mục gốc:

```bash
python - <<'EOF'
import sys
sys.path.insert(0, "/tmp")          # nơi lưu sub.py theo Phụ lục A
from sub import sub

sub("README.md",
    "*(Tạo phân chia stratified 60/20/20 cho cả bộ dữ liệu đầy đủ và bộ dữ liệu đã khử trùng lặp)*",
    "*(Tạo phân chia phân tầng theo nhãn 3 lớp: train 68% / val 12% / test 20% — val là 15% của phần còn lại sau khi tách test, seed 42 — cho cả bộ đầy đủ và bộ đã khử trùng lặp)*")
sub("README.md",
    "Loại bỏ 24.206 bản ghi có vector đặc trưng trùng lặp hoàn toàn khỏi dữ liệu gốc:",
    "Loại bỏ 23.899 bản ghi trùng lặp hoàn toàn (cả 21 đặc trưng và nhãn), còn 229.781 bản ghi:")

sub("docs/architecture.md", "(`stratified_split`)", "(`make_split`)")
sub("docs/architecture.md", "3 lớp với Dropout/BatchNorm", "21→128→64→32→3 với ReLU và Dropout")
sub("docs/architecture.md", "`/predict`, `/explain/global`, `/health`", "`/predict`, `/explain`, `/model-info`, `/health`")
sub("docs/architecture.md",
    "Tập dữ liệu đã khử bỏ 24.206 dòng trùng lặp hoàn toàn (còn lại 229.474 bản ghi).",
    "Tập dữ liệu đã khử bỏ 23.899 dòng trùng lặp hoàn toàn (còn lại 229.781 bản ghi).")

sub("reports/final_report.md",
    "đã được thực hiện trên tập dữ liệu kiểm thử.",
    "đã được thực hiện trên tập validation (không dùng tập test).")
EOF
```

Nếu một lệnh `sub` báo `tìm thấy 0 vị trí`, mở file, `grep -n` đoạn tương ứng và sửa **đúng nghĩa** theo bảng trên (không bỏ qua). Sau đó chạy `grep -n "kiểm thử" reports/model_card.md reports/final_report.md reports/fairness_report.md` và kiểm tra từng nơi nhắc permutation/giải thích có nói "test" không; nếu có, sửa thành validation.

**Bước F07.2 — bảo đảm mục "Hạn chế" của `reports/final_report.md` (§8, ngoài các khối `<!-- AUTO ... -->`) có hai ý sau; nếu thiếu thì thêm** (đọc số từ `reports/metrics/test_evaluation.json → multiclass_full → mlp → risk_alert_stats` để chắc chắn khớp):
- Tỉ lệ cảnh báo: ở ngưỡng "trung bình" (chỉ số ≥ 0,10) khoảng **49,9%** người được xếp từ mức trung bình trở lên (PPV 25,3%, độ nhạy 90,4%); ở mức "cao" (≥ 0,25) là **22,2%** (PPV 38,0%, độ nhạy 60,5%). Đây là đánh đổi có chủ đích của công cụ **sàng lọc**, không phải chẩn đoán.
- Lớp "tiền tiểu đường" gần như không phân biệt được (recall test 4,5%, precision 4,1%).

**Bước F07.3 — test chống lệch số liệu.** Tạo `tests/test_docs_consistency.py`:

```python
"""Tài liệu phải dùng đúng số đo, không còn số cũ/sai."""

import json

from diabetes_ai.paths import METRICS_DIR, ROOT


def _vn(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_docs_do_not_contain_stale_numbers() -> None:
    for rel in ("README.md", "docs/architecture.md"):
        text = _read(rel)
        assert "24.206" not in text, f"{rel} còn số trùng lặp cũ 24.206"
        assert "229.474" not in text, f"{rel} còn số 229.474"
    assert "60/20/20" not in _read("README.md")
    assert "/explain/global" not in _read("docs/architecture.md")
    assert "BatchNorm" not in _read("docs/architecture.md")


def test_docs_use_measured_duplicate_counts() -> None:
    val = json.loads((METRICS_DIR / "data_validation.json").read_text(encoding="utf-8"))
    arch = _read("docs/architecture.md")
    assert _vn(val["n_duplicates"]) in arch
    assert _vn(val["n_unique_rows"]) in arch
    assert _vn(val["n_duplicates"]) in _read("README.md")
```

**Kiểm chứng F07:**

```bash
python -m pytest -q tests/test_docs_consistency.py tests/test_docs_disclaimer.py
python scripts/make_reports.py --check     # thoát 0: các khối AUTO vẫn đầy
git diff --stat                            # chỉ vài dòng thay đổi mỗi file (không phải cả file → CRLF được giữ)
```

**Commit:** `docs: sửa số liệu và mô tả lệch với mã (chia 68/12/20, 23.899 dòng trùng, endpoint, MLP)`

---

### F08 — Nối hệ đa agent vào API (`POST /consult`) và giao diện

**Bằng chứng:** `agents/orchestrator.py::Orchestrator.consult(profile: dict) -> dict` có sẵn, có test, trả `events` (chuỗi `handoff/tool/speak/alert/cite/final`), `risk_level`, `final_level`, `risk_index`, `probabilities`, `alerts`, `citations`, `advice`, `disclaimer`. Nhưng không có tham chiếu nào trong `serving/` hay `app/` (`grep -rn "orchestrator\|Orchestrator" src/diabetes_ai/serving app configs` rỗng). Làm **sau F03 và F04**.

**Thiết kế:** thêm endpoint `POST /consult` nhận đúng `PatientProfile` như `/predict`; dùng chung một `Predictor` (singleton) cho `RiskAgent` để không nạp mô hình hai lần; giữ nguyên `/predict`, `/explain`.

**Bước F08.1 — `src/diabetes_ai/inference/schemas.py`:** thêm sau lớp `ExplainResponse`:

```python
class ConsultResponse(BaseModel):
    """Lược đồ phản hồi tư vấn đa tác tử (Risk → Safety → Knowledge)."""

    events: list[dict[str, Any]]
    risk_level: str
    final_level: str
    risk_index: float
    probabilities: dict[str, float]
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)
    disclaimer: str
```

**Bước F08.2 — `src/diabetes_ai/serving/deps.py`:** thêm import và singleton; `reset_predictor` phải reset cả hai.

```python
from diabetes_ai.agents.orchestrator import Orchestrator
from diabetes_ai.agents.risk_agent import RiskAgent
```

```python
_ORCHESTRATOR: Orchestrator | None = None


def reset_predictor() -> None:
    """Xóa bỏ Predictor và Orchestrator hiện tại để buộc nạp lại (dùng trong kiểm thử)."""
    global _PREDICTOR, _ORCHESTRATOR
    _PREDICTOR = None
    _ORCHESTRATOR = None


def get_orchestrator() -> Orchestrator:
    """Lấy singleton Orchestrator dùng chung Predictor đã nạp (503 nếu mô hình chưa sẵn sàng)."""
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = Orchestrator(risk_agent=RiskAgent(predictor=get_predictor()))
    return _ORCHESTRATOR
```

(thay hàm `reset_predictor` hiện có bằng bản trên; không đổi `get_predictor`.)

**Bước F08.3 — `src/diabetes_ai/serving/routes.py`:** thêm import `ConsultResponse` vào khối import từ `schemas`, import `Orchestrator` và `get_orchestrator`, rồi thêm route:

```python
from diabetes_ai.agents.orchestrator import Orchestrator
from diabetes_ai.serving.deps import get_orchestrator, get_predictor
```

```python
@router.post("/consult", response_model=ConsultResponse)
def consult(
    profile: PatientProfile,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ConsultResponse:
    """Tư vấn đa tác tử: RiskAgent → SafetyAgent (luật + phản biện) → KnowledgeAgent (trích dẫn có nguồn)."""
    return ConsultResponse(**orchestrator.consult(profile.model_dump()))
```

(đổi dòng `from diabetes_ai.serving.deps import get_predictor` hiện có thành dòng import hai tên ở trên.)

**Bước F08.4 — test** (thêm vào cuối `tests/test_api_predict.py`):

```python
def test_consult_ends_with_final_and_has_disclaimer(client: TestClient, sample_profile: dict) -> None:
    res = client.post("/consult", json=sample_profile)
    assert res.status_code == 200
    data = res.json()
    assert data["events"][-1]["event"] == "final"
    assert data["risk_level"] in {"thấp", "trung bình", "cao"}
    assert data["disclaimer"]
    assert any("xét nghiệm xác nhận" in a.lower() for a in data["advice"])


def test_consult_danger_alert_forces_high_level(client: TestClient, sample_profile: dict) -> None:
    payload = {**sample_profile, "glucose": 280.0}
    data = client.post("/consult", json=payload).json()
    assert data["final_level"] == "cao"


def test_consult_rejects_invalid_profile_422(client: TestClient, sample_profile: dict) -> None:
    res = client.post("/consult", json={**sample_profile, "BMI": 5})
    assert res.status_code == 422
```

**Bước F08.5 — giao diện (tùy chọn nhưng khuyến nghị).**
(a) `app/index.html`: chèn **ngay trước** dòng `<div class="disclaimer-bottom" role="note">` (khoảng dòng 350):

```html
        <div id="consult-block" class="whatif-section">
          <h3>Tư vấn đa tác tử (có trích dẫn)</h3>
          <button type="button" id="consult-btn">Xem tư vấn có trích dẫn</button>
          <div id="consult-output" aria-live="polite"></div>
        </div>
```

(b) `app/static/js/app.js`: thêm hàm (đặt trước `// Lắng nghe sự kiện`) và gắn sự kiện trong `DOMContentLoaded` (cạnh các `addEventListener` hiện có). Dùng `textContent`/DOM, không dùng `innerHTML`:

```javascript
async function callConsult() {
  const out = document.getElementById("consult-output");
  out.textContent = "";
  if (!currentProfile) {
    out.textContent = "Hãy bấm “Đánh giá nguy cơ” trước.";
    return;
  }
  try {
    const res = await fetch(`${API}/consult`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentProfile),
    });
    if (!res.ok) {
      out.textContent = "Không lấy được tư vấn (mã lỗi " + res.status + ").";
      return;
    }
    const data = await res.json();
    const head = document.createElement("p");
    head.textContent = `Mức nguy cơ sau rà soát an toàn: ${data.final_level}`;
    out.appendChild(head);

    const ul = document.createElement("ul");
    (data.citations || []).forEach((c) => {
      const li = document.createElement("li");
      const b = document.createElement("strong");
      b.textContent = c.title || "Tài liệu";
      li.append(b, document.createTextNode(` — ${c.source || ""}`));
      ul.appendChild(li);
    });
    out.appendChild(ul);

    const note = document.createElement("p");
    note.textContent = data.disclaimer || "";
    out.appendChild(note);
  } catch (err) {
    out.textContent = "Không kết nối được máy chủ.";
  }
}
```

và trong `DOMContentLoaded`: `document.getElementById("consult-btn").addEventListener("click", callConsult);`

(c) Test tĩnh: thêm vào `tests/test_demo_static.py`:

```python
def test_app_js_calls_consult_endpoint() -> None:
    js = (ROOT / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "/consult" in js
```

**Bước F08.6 — tài liệu.** Thêm `/consult` (và `/model-info`) vào danh sách endpoint của `README.md` và dòng `Serving` trong `docs/architecture.md` (nếu F07 đã chạy thì dòng đó đang là `` `/predict`, `/explain`, `/model-info`, `/health` `` → thêm `` `/consult` ``). Cập nhật test `tests/test_docs_consistency.py` nếu cần (không có kiểm tra nào chặn `/consult`).

**Kiểm chứng F08:**

```bash
python -m pytest -q tests/test_api_predict.py tests/test_agents.py tests/test_demo_static.py
python - <<'EOF'
import json, subprocess, sys, time, httpx
p = subprocess.Popen([sys.executable, "-m", "uvicorn", "diabetes_ai.serving.app:app", "--port", "8200"])
try:
    for _ in range(40):
        try: httpx.get("http://127.0.0.1:8200/health", timeout=1); break
        except Exception: time.sleep(0.5)
    prof = json.load(open("tests/fixtures/sample_profile.json", encoding="utf-8"))
    r = httpx.post("http://127.0.0.1:8200/consult", json=prof, timeout=30)
    d = r.json()
    print(r.status_code, d["events"][-1]["event"], d["final_level"], len(d["citations"]))
finally:
    p.terminate(); p.wait(timeout=10)
EOF
```

Kỳ vọng dòng in: `200 final <thấp|trung bình|cao> <số ≥ 0>`.

**Rủi ro:** `KnowledgeAgent`/`graph_rag.search` đọc `knowledge_base/processed/chunks.json` mỗi lần gọi; chấp nhận được cho demo. Nếu đo thấy chậm (> 1 giây), bọc việc nạp JSON bằng `functools.lru_cache` trong `knowledge/graph_rag.py` (không đổi chữ ký).

**Commit:** `feat(api): endpoint /consult nối hệ đa agent và nút tư vấn có trích dẫn trên giao diện`

---

### F09 — Hạ tầng: `Makefile`, Docker, CI

**Bằng chứng:**
- `Makefile`: dòng `all: bash scripts/run_all.sh` khai báo `bash` và `scripts/run_all.sh` là **điều kiện tiên quyết** của target `all`, không phải lệnh → `make all` không chạy `run_all.sh` (và báo thiếu quy tắc tạo `bash`).
- `Dockerfile` cài toàn bộ `requirements.txt` (kèm `torch`, `shap`, `matplotlib`) dù máy chủ chạy bundle NumPy; chạy bằng root; không `HEALTHCHECK`; `COPY . .` kéo cả `tests/`, `reports/`, `docs/`.
- `docker-compose.yml`: `env_file: .env` **bắt buộc** — thiếu `.env` thì `docker compose up` báo lỗi.
- `.github/workflows/ci.yml`: không cache pip; cài `torch` bản mặc định (rất nặng); chỉ Python 3.11 (người dùng chạy 3.14); không `ruff format --check` dù `Makefile lint` có.

**Bước F09.1 — `Makefile`:** thay

```make
all: bash scripts/run_all.sh
```

bằng hai dòng sau (**dòng thứ hai bắt đầu bằng ký tự TAB**, không phải dấu cách — Makefile hỏng nếu dùng dấu cách):

```make
all:
	bash scripts/run_all.sh
```

Kiểm tra: `make -n all` (nếu có `make`) in `bash scripts/run_all.sh`.

**Bước F09.2 — tạo `requirements-serve.txt`** (chỉ phụ thuộc cần để chạy API với bundle NumPy):

```
numpy
pandas
scikit-learn
joblib
pyyaml
python-dotenv
pydantic
fastapi
uvicorn[standard]
```

**Xác minh bộ tối thiểu bằng venv sạch** (bắt buộc, vì chưa kiểm chứng được `import` nào kéo thêm thư viện):

```bash
python -m venv /tmp/venv-serve && source /tmp/venv-serve/Scripts/activate   # Linux/macOS: .../bin/activate
pip install -r requirements-serve.txt pytest httpx
pip install -e . --no-deps
python -c "from diabetes_ai.serving.app import app; print('import OK')"
python -m pytest -q tests/test_api_predict.py tests/test_api_health.py
deactivate
```

Nếu `import` báo thiếu `xgboost`, thêm `xgboost` vào `requirements-serve.txt`. Nếu thiếu `torch` hoặc `matplotlib` → **dừng và báo lại** (cần chuyển import xuống bên trong hàm ở `models/registry.py` hoặc nơi liên quan; việc đó nằm ngoài mục này).

**Bước F09.3 — `Dockerfile`** (thay toàn bộ):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY app ./app
COPY knowledge_base ./knowledge_base
RUN pip install --no-cache-dir -e . --no-deps
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8200
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8200/health', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "diabetes_ai.serving.app:app", "--host", "0.0.0.0", "--port", "8200"]
```

M� hình **không** nằm trong image (git bỏ qua `*.joblib`); compose gắn thư mục `models/` vào `/app/models`.

**Bước F09.4 — `docker-compose.yml`** (thay toàn bộ):

```yaml
services:
  api:
    build: .
    ports: ["8200:8200"]
    environment:
      MODEL_PATH: ${MODEL_PATH:-models/best_model.joblib}
    volumes:
      - ./models:/app/models:ro
```

**Bước F09.5 — `.dockerignore`** (thay toàn bộ):

```
.git
.github
.venv
data
mlruns
__pycache__
notebooks
tests
reports
docs
models
experiments
```

(không loại `README.md` vì `pyproject.toml` tham chiếu `readme = "README.md"`.)

**Bước F09.6 — `.github/workflows/ci.yml`** (thay toàn bộ):

```yaml
name: CI
on:
  push: { branches: [main] }
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: requirements*.txt
      - name: Cài torch bản CPU (nhẹ)
        run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -r requirements-dev.txt && pip install -e .
      - run: ruff check src tests
      - run: pytest
```

Trước khi thêm bước `ruff format --check src tests` vào CI: chạy cục bộ `python -m ruff format --check src tests`. Nếu báo cần định dạng lại, chạy `python -m ruff format src tests`, kiểm tra `git diff --stat` rồi **commit riêng** `style: ruff format` (chỉ định dạng, không đổi logic; chạy lại `pytest`); sau đó mới thêm `- run: ruff format --check src tests` vào CI.

**Bước F09.7 (tùy chọn) — `pyproject.toml`:** để `pip install .` tự kéo phụ thuộc phục vụ, trong `[project]` thêm `dynamic = ["dependencies"]` và thêm bảng:

```toml
[tool.setuptools.dynamic]
dependencies = { file = ["requirements-serve.txt"] }
```

Kiểm tra `pip install -e . --dry-run` không lỗi. Bỏ bước này nếu gặp vấn đề với phiên bản `setuptools`.

**Kiểm chứng F09:**

```bash
python -m ruff check src tests
python -m pytest -q
docker build -t diabetes-ai-lab . && docker run --rm -p 8200:8200 -v "$PWD/models:/app/models:ro" diabetes-ai-lab &
sleep 15 && curl -s http://127.0.0.1:8200/health      # {"status":"ok"}
```

(chỉ chạy hai dòng Docker nếu máy có Docker; nếu không, ghi rõ "chưa kiểm chứng Docker" trong báo cáo.)

**Commit:** `chore(infra): sửa Makefile, Dockerfile nhẹ, compose không cần .env, CI cache và ma trận Python`

---

### F10 — Gói mô hình không ghi phiên bản thư viện; `export_summary.json` chứa đường dẫn tuyệt đối

**Bằng chứng:** nạp thật `models/best_model.joblib` cho thấy các khóa `['backend','calibrator','class_names','disclaimer','feature_order','metrics_val','model','pop_mean_raw','risk_levels','run_name','scale_factors','scaler','schema_version','task','threshold','trained_on','variant','what_if']` — không có phiên bản. `calibrator` là đối tượng scikit-learn (pickle): khác phiên bản scikit-learn khi nạp có thể cảnh báo hoặc lỗi âm thầm. `reports/metrics/export_summary.json` chứa `"path": "F:\\PT các HTTM\\AITRAINING\\diabetes-ai-lab\\models\\best_model.joblib"` (đường dẫn máy cá nhân, không tái lập).

**Bước F10.0 — chụp kết quả trước khi đổi** (để chứng minh không đổi số):

```bash
python - <<'EOF'
import json
from diabetes_ai.inference.predictor import Predictor
prof = json.load(open("tests/fixtures/sample_profile.json", encoding="utf-8"))
json.dump(Predictor.from_bundle().predict(prof)["probabilities"], open("/tmp/before.json", "w"), ensure_ascii=False)
EOF
```

**Bước F10.1 — `src/diabetes_ai/inference/export.py`.**
(a) Thêm hàm ngay **trước** `def make_bundle(`:

```python
def _runtime_env() -> dict[str, str | None]:
    """Phiên bản Python và thư viện tại thời điểm đóng gói (để cảnh báo khi nạp lệch phiên bản)."""
    import platform
    from importlib import metadata

    def _ver(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "numpy": _ver("numpy"),
        "scikit-learn": _ver("scikit-learn"),
        "joblib": _ver("joblib"),
        "xgboost": _ver("xgboost"),
    }
```

(b) Trong `make_bundle`, ngay sau dòng `"schema_version": 1,` (giữ nguyên giá trị `1`) thêm: `"env": _runtime_env(),`.

**Bước F10.2 — `src/diabetes_ai/inference/predictor.py`.** Thêm `import warnings` ở đầu file; thêm hàm cấp module (trước lớp `Predictor`):

```python
def _warn_if_env_mismatch(bundle: dict[str, Any]) -> None:
    """Cảnh báo khi phiên bản numpy/scikit-learn lúc nạp khác lúc đóng gói (pickle có thể không tương thích)."""
    from importlib import metadata

    saved = bundle.get("env") or {}
    for pkg in ("numpy", "scikit-learn"):
        old = saved.get(pkg)
        try:
            current = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
        if old and old.split(".")[:2] != current.split(".")[:2]:
            warnings.warn(
                f"Gói mô hình được đóng gói với {pkg}=={old} nhưng môi trường hiện tại là {current}; "
                "nếu gặp lỗi nạp hoặc kết quả lạ, hãy chạy lại scripts/export_model.py.",
                RuntimeWarning,
                stacklevel=2,
            )
```

Trong `Predictor.from_bundle`, ngay sau `bundle = load_bundle(target_path)` thêm `_warn_if_env_mismatch(bundle)`. Bundle cũ không có khóa `env` thì hàm không làm gì (tương thích ngược).

**Bước F10.3 — `scripts/export_model.py`:** đổi `from diabetes_ai.paths import METRICS_DIR, ROOT` thành `from diabetes_ai.paths import METRICS_DIR, ROOT, rel` và trong dict `summary` đổi `"path": str(saved_path),` thành `"path": rel(saved_path),`.

**Bước F10.4 — test** (thêm vào `tests/test_predictor.py`; thêm `import pytest` và `from diabetes_ai.inference.export import load_bundle, save_bundle` nếu chưa có):

```python
def test_bundle_records_runtime_env(tiny_bundle) -> None:
    env = load_bundle(tiny_bundle)["env"]
    assert env["python"] and env["numpy"] and env["scikit-learn"]


def test_predictor_warns_on_sklearn_version_mismatch(tiny_bundle, tmp_path) -> None:
    bundle = load_bundle(tiny_bundle)
    bundle["env"]["scikit-learn"] = "0.0.1"
    path = tmp_path / "mismatch.joblib"
    save_bundle(bundle, path)
    with pytest.warns(RuntimeWarning):
        Predictor.from_bundle(path)
```

**Bước F10.5 — đóng gói lại (không huấn luyện) và so sánh:**

```bash
python scripts/export_model.py
python - <<'EOF'
import json
from diabetes_ai.inference.predictor import Predictor
prof = json.load(open("tests/fixtures/sample_profile.json", encoding="utf-8"))
after = Predictor.from_bundle().predict(prof)["probabilities"]
before = json.load(open("/tmp/before.json"))
print(max(abs(after[k] - before[k]) for k in before))
EOF
cat reports/metrics/export_summary.json
```

Kỳ vọng: sai khác tối đa `< 1e-6`; `export_summary.json` có `"path": "models/best_model.joblib"` (dấu `/`) và `bytes` ≈ 62 KB; `sha256` đổi (bình thường). `git status` chỉ thay đổi `reports/metrics/export_summary.json` (và code/test); `models/best_model.joblib` vẫn bị git bỏ qua.

**Commit:** `feat(export): ghi phiên bản thư viện vào bundle, cảnh báo lệch phiên bản; đường dẫn tương đối trong export_summary`

---

## 5. Nhóm P2 — làm chắc, vận hành, tái lập

### F11 — Giao diện dùng `innerHTML` với dữ liệu từ máy chủ

**Bằng chứng:** `app/static/js/app.js` dựng HTML bằng chuỗi: `showError` (ghép `message`, `err.field`, `err.message` rồi gán `innerHTML`) và `renderDrivers` (`div.innerHTML` chứa `d.vi`, `d.feature`, `d.value`). Hiện dữ liệu do máy chủ của chính dự án sinh nên rủi ro thấp, nhưng nếu thông điệp lỗi sau này phản chiếu đầu vào người dùng thì thành lỗ hổng XSS. Các chỗ gán `innerHTML = ""` (xóa nội dung) không cần sửa.

**Bước F11.1 — thay hàm `showError`:**

```javascript
function showError(message, fieldErrors = []) {
  const errorContainer = document.getElementById("error-container");
  errorContainer.style.display = "block";

  let html = `<strong>Đã xảy ra lỗi:</strong> <span>${message}</span>`;
  if (fieldErrors.length > 0) {
    html += "<ul>";
    fieldErrors.forEach((err) => {
      html += `<li><strong>${err.field}:</strong> ${err.message}</li>`;
    });
    html += "</ul>";
  }
  errorContainer.innerHTML = html;
  errorContainer.scrollIntoView({ behavior: "smooth" });
}
```

bằng

```javascript
function showError(message, fieldErrors = []) {
  const errorContainer = document.getElementById("error-container");
  errorContainer.style.display = "block";
  errorContainer.textContent = "";

  const title = document.createElement("strong");
  title.textContent = "Đã xảy ra lỗi: ";
  const msg = document.createElement("span");
  msg.textContent = String(message);
  errorContainer.append(title, msg);

  if (fieldErrors.length > 0) {
    const ul = document.createElement("ul");
    fieldErrors.forEach((err) => {
      const li = document.createElement("li");
      const f = document.createElement("strong");
      f.textContent = `${err.field}: `;
      li.append(f, document.createTextNode(String(err.message)));
      ul.appendChild(li);
    });
    errorContainer.appendChild(ul);
  }
  errorContainer.scrollIntoView({ behavior: "smooth" });
}
```

**Bước F11.2 — trong `renderDrivers`,** thay

```javascript
    div.innerHTML = `
      <span class="driver-title">${d.vi || d.feature}</span>
      <div>Giá trị hiện tại: <strong>${d.value}</strong></div>
      <span class="driver-effect ${effectClass}">
        ${sign}${(d.effect * 100).toFixed(2)}% (${dirText})
      </span>
    `;
    driversList.appendChild(div);
```

bằng

```javascript
    const title = document.createElement("span");
    title.className = "driver-title";
    title.textContent = d.vi || d.feature;

    const current = document.createElement("div");
    current.append("Giá trị hiện tại: ");
    const val = document.createElement("strong");
    val.textContent = String(d.value);
    current.appendChild(val);

    const effect = document.createElement("span");
    effect.className = `driver-effect ${effectClass}`;
    effect.textContent = `${sign}${(d.effect * 100).toFixed(2)}% (${dirText})`;

    div.append(title, current, effect);
    driversList.appendChild(div);
```

**Bước F11.3 — CORS: không đổi hành vi.** `app.js` đặt `API = "http://localhost:8200"` khi trang mở bằng `file://`, nên origin `"null"` trong `configs/serve.yaml` là **có chủ đích** (cho phép mở `app/index.html` trực tiếp). Chỉ thêm chú thích trên dòng `cors_origins:` trong `configs/serve.yaml`:

```yaml
# "null" cho phép mở app/index.html bằng file://; xóa dòng này khi triển khai công khai.
cors_origins:
```

**Bước F11.4 — test tĩnh** (thêm vào `tests/test_demo_static.py`):

```python
def test_app_js_has_no_dynamic_inner_html() -> None:
    """Chỉ cho phép innerHTML để xóa nội dung (gán chuỗi rỗng); không ghép HTML từ dữ liệu."""
    js = (ROOT / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    for line in js.splitlines():
        if "innerHTML" in line:
            assert '= ""' in line, f"innerHTML động không được phép: {line.strip()}"
```

**Kiểm chứng:** `python -m pytest -q tests/test_demo_static.py`; thử tay: gửi form BMI = 5 → khung lỗi hiện thông điệp bình thường (không méo định dạng).

**Commit:** `fix(demo): dựng DOM bằng textContent thay vì innerHTML`

---

### F12 — Đọc cấu hình mỗi request; `/health` không phản ánh trạng thái mô hình

**Bằng chứng:** `Predictor.explain` gọi `load_config("features")` (đọc và phân tích YAML) **mỗi request** để lấy `drop_for_what_if`. `/health` luôn trả `{"status": "ok"}` kể cả khi chưa nạp được mô hình (yêu cầu giữ nguyên theo test), nên thiếu cách kiểm tra "sẵn sàng".

**Bước F12.1 — `src/diabetes_ai/inference/predictor.py`.** Thêm `from functools import lru_cache` và hàm cấp module (trước lớp `Predictor`):

```python
@lru_cache(maxsize=1)
def _drop_for_what_if() -> tuple[str, ...]:
    """Đọc một lần danh sách đặc trưng không đưa vào what-if từ configs/features.yaml."""
    try:
        return tuple(load_config("features").get("drop_for_what_if", []))
    except Exception:
        return ()
```

Trong `explain`, thay

```python
        try:
            feat_cfg = load_config("features")
            drop_list = set(feat_cfg.get("drop_for_what_if", []))
        except Exception:
            drop_list = set()
```

bằng `drop_list = set(_drop_for_what_if())`.

**Bước F12.2 — `src/diabetes_ai/serving/app.py`.** Ngay sau hàm `health()` (không đổi `/health`) thêm:

```python
@app.get("/ready")
def ready() -> dict[str, str]:
    """Kiểm tra sẵn sàng: 200 khi mô hình đã nạp được, 503 (MODEL_NOT_READY) khi chưa."""
    get_predictor()  # ném HTTPException 503 nếu chưa sẵn sàng
    return {"status": "ready"}
```

**Bước F12.3 — test** (thêm vào `tests/test_api_predict.py`; `reset_predictor` đã được import ở đầu file):

```python
def test_ready_ok_when_model_loaded(client: TestClient) -> None:
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json() == {"status": "ready"}


def test_ready_503_when_model_missing(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "khong_ton_tai.joblib"))
    reset_predictor()
    res = client.get("/ready")
    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "MODEL_NOT_READY"
```

**Kiểm chứng:** `python -m pytest -q tests/test_api_predict.py tests/test_explain.py tests/test_predictor.py`; `test_health_unchanged` vẫn pass.

**Commit:** `perf(api): cache cấu hình what-if; feat(api): thêm /ready`

---

### F13 — `run_all.sh` chưa kiểm tra khối AUTO của báo cáo

**Bằng chứng:** plan K10 yêu cầu `python scripts/make_reports.py --check` thoát 0; `make_reports.py` đã có cờ `--check` (thoát 1 nếu còn khối AUTO trống) nhưng `scripts/run_all.sh` chỉ gọi `python scripts/make_reports.py`.

**Bước F13.1 — `scripts/run_all.sh`:** ngay dưới dòng `python scripts/make_reports.py` (bước 8) thêm một dòng:

```bash
python scripts/make_reports.py --check
```

**Kiểm chứng (không chạy cả pipeline):** `bash -n scripts/run_all.sh` (không báo lỗi cú pháp) và `python scripts/make_reports.py --check; echo $?` → `0`. **Không** chạy `run_all.sh` đầy đủ.

**Commit:** `chore(pipeline): run_all.sh kiểm tra khối AUTO sau khi sinh báo cáo`

---

### F14 — `env_check.json` không ghi phiên bản gói

**Bằng chứng:** `reports/metrics/env_check.json` chỉ có `python`, `platform` và cờ `*_ok`. Không có phiên bản numpy/scikit-learn/torch/xgboost/fastapi/pydantic, nên không thể tái lập môi trường tạo ra kết quả.

**Bước F14.1 — `scripts/check_env.py`.** Thêm hàm ngay trước `def main()`:

```python
def _package_versions() -> dict[str, str | None]:
    """Phiên bản các gói chính (None nếu chưa cài)."""
    from importlib import metadata

    names = ["numpy", "pandas", "scikit-learn", "xgboost", "torch", "shap", "fastapi", "pydantic", "uvicorn", "joblib", "pyyaml"]
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out
```

Trong dict `report_payload` (nơi có `"core_ok": core_ok,`) thêm khóa `"packages": _package_versions(),` (đặt trước khóa `csv` hoặc cuối dict; không đổi các khóa hiện có).

**Kiểm chứng:** `python scripts/check_env.py` thoát 0 và `reports/metrics/env_check.json` có mục `packages`. Các khóa cũ (`core_ok`, `csv.sha256`…) giữ nguyên giá trị (`csv.sha256` vẫn `a971809d…`).

**Commit:** `chore(env): ghi phiên bản gói vào env_check.json`

---

### F15 — Cách đọc `label` và tỉ lệ cảnh báo chưa được nêu rõ

**Bằng chứng:** hệ số ngưỡng của gói mô hình là `scale_factors = [1.0, 17.98, 2.38]` (nạp thật từ bundle). `Predictor.predict` chọn `label = argmax(xác suất × hệ số)`, nên `label` có thể là "tiền tiểu đường" dù P(tiền tiểu đường) chỉ vài phần trăm (hệ số tối ưu Macro-F1 trên validation, ưu tiên độ nhạy). Giao diện chỉ hiển thị xác suất và mức nguy cơ (không hiển thị `label`) nên người dùng web không bị nhầm, nhưng khách gọi API có thể hiểu sai. Tỉ lệ cảnh báo (≈ 49,9% từ mức trung bình trở lên) chưa nằm trong tài liệu (xem F07.2).

**Bước F15.1 — `src/diabetes_ai/inference/schemas.py`,** trong `PredictionResponse`, thay dòng `    label: str` (nằm giữa `risk_level: str` và `alerts`) bằng:

```python
    label: str = Field(
        description=(
            "Nhãn theo ngưỡng sàng lọc (xác suất nhân hệ số để ưu tiên độ nhạy); có thể là "
            "'tiền tiểu đường' dù xác suất lớp này thấp. Hãy dựa vào `probabilities` và `risk_level`."
        )
    )
```

**Bước F15.2 — tài liệu.** Thêm mục ngắn "Cách đọc kết quả" vào `README.md` (sau phần API) và vào mục hạn chế của `reports/model_card.md` (ngoài khối AUTO): (1) `risk_level` và `probabilities` là thông tin chính, `label` chỉ mang tính tham khảo ngưỡng; (2) chỉ số nguy cơ `r = P(tiểu đường) + 0,5·P(tiền tiểu đường)`, ngưỡng 0,10 và 0,25; (3) nhóm "trung bình trở lên" chiếm khoảng một nửa số người (đánh đổi của công cụ sàng lọc); (4) mọi kết quả cần xét nghiệm xác nhận. Không thêm số mới ngoài các số đã nêu ở F07.2.

**Kiểm chứng:** `python -m pytest -q tests/test_docs_disclaimer.py tests/test_schemas.py tests/test_api_predict.py`; mở `http://localhost:8200/docs` → mô tả của trường `label` hiển thị.

**Commit:** `docs(api): giải thích cách đọc label và tỉ lệ cảnh báo`

---

## 6. Nghiệm thu cuối (làm sau khi xong mọi mục)

Chạy lần lượt và ghi kết quả vào báo cáo cho người dùng:

```bash
# 1. Toàn bộ test (ghi lại số passed; so với kết quả B0)
python -m pytest -q 2>&1 | tail -15
python -m ruff check src tests

# 2. Test trên bản clone sạch (mô phỏng CI)
rm -rf /tmp/dal-clean && git clone . /tmp/dal-clean && (cd /tmp/dal-clean && python -m pytest -q 2>&1 | tail -15)

# 3. Kết quả đã khóa không bị đụng tới
git diff --stat HEAD~N -- reports/metrics/test_evaluation.json reports/metrics/test_lock.json   # N = số commit của đợt sửa; kỳ vọng: rỗng
grep -rn "include_test=True" src scripts      # kỳ vọng: chỉ scripts/evaluate.py

# 4. Báo cáo tự sinh vẫn đầy đủ
python scripts/make_reports.py --check; echo $?    # 0

# 5. Smoke test thật (tự khởi động và tắt máy chủ)
python - <<'EOF'
import json, subprocess, sys, time, httpx
base = "http://127.0.0.1:8200"
p = subprocess.Popen([sys.executable, "-m", "uvicorn", "diabetes_ai.serving.app:app", "--port", "8200"])
try:
    for _ in range(40):
        try: httpx.get(f"{base}/health", timeout=1); break
        except Exception: time.sleep(0.5)
    prof = json.load(open("tests/fixtures/sample_profile.json", encoding="utf-8"))
    ui = {"BMI": 24.5, "Smoker": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0}
    print("health ", httpx.get(f"{base}/health").json())
    print("ready  ", httpx.get(f"{base}/ready").status_code)
    print("predict", httpx.post(f"{base}/predict", json=prof).status_code,
          httpx.post(f"{base}/predict", json={**prof, "BMI": 5}).status_code)
    print("explain", httpx.post(f"{base}/explain", json={"profile": prof, "what_if_changes": ui}).status_code,
          httpx.post(f"{base}/explain", json={"profile": prof, "what_if_changes": {"HighBP": 0}}).status_code)
    print("consult", httpx.post(f"{base}/consult", json=prof, timeout=30).status_code)
    print("app    ", "Chỉ để học tập" in httpx.get(f"{base}/app/").text)
finally:
    p.terminate(); p.wait(timeout=10)
EOF
```

Kết quả mong đợi của bước 5:

```
health  {'status': 'ok'}
ready   200
predict 200 422
explain 200 422
consult 200
app     True
```

**Checklist hoàn tất** (đánh dấu từng dòng, dán vào báo cáo cuối):

| Mục | Điều kiện đạt |
|---|---|
| F01 | UI chỉ gửi 6 khóa hợp lệ; test hợp đồng pass; kéo thanh BMI cập nhật được bảng what-if |
| F02 | `/explain` trả 422 cho khóa lạ, ngoài miền, `NaN`; không còn 500 |
| F03 | `SafetyAgent` dùng đúng P(tiểu đường); test R10 pass |
| F04 | `pytest` pass trên bản clone sạch và khi không có `models/best_model.joblib` |
| F05 | 7 file `eda_*.png`, notebook 18 ô chạy hết, `data_dictionary.md` đủ số |
| F06 | `.env` có hiệu lực; `.env.example` gọn; test pass |
| F07 | không còn `24.206`, `229.474`, `60/20/20`, `/explain/global`, `BatchNorm`; permutation ghi "validation" |
| F08 | `/consult` hoạt động; nút tư vấn trên giao diện; test pass |
| F09 | `Makefile`, Docker, compose, CI đã sửa; `requirements-serve.txt` đã xác minh bằng venv sạch |
| F10 | bundle có `env`; sai khác xác suất < 1e-6; `export_summary.json` dùng đường dẫn tương đối |
| F11–F15 | test tương ứng pass; không đổi khóa JSON cũ |

**Nếu một mục không làm được** (ví dụ không có Docker, hoặc `requirements-serve.txt` thiếu thư viện), ghi rõ lý do và phần chưa kiểm chứng; không đánh dấu hoàn tất.

---

## Phụ lục A — hàm `sub()` để sửa file giữ nguyên CRLF

Lưu thành `/tmp/sub.py` (công cụ tạm, **không commit**). Hàm đọc/ghi dạng bytes nên không làm đổi kết thúc dòng; nếu file dùng CRLF thì tự đổi `old`/`new` sang CRLF trước khi so khớp, và dừng nếu số vị trí khớp khác mong đợi:

```python
from pathlib import Path


def sub(path: str, old: str, new: str, count: int = 1) -> None:
    p = Path(path)
    s = p.read_bytes().decode("utf-8")
    if "\r\n" in s:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    n = s.count(old)
    assert n == count, f"{path}: tìm thấy {n} vị trí (cần {count}) cho đoạn: {old[:120]!r}"
    p.write_bytes(s.replace(old, new).encode("utf-8"))
```

Cách dùng: `python - <<'EOF'` … `import sys; sys.path.insert(0, "/tmp"); from sub import sub` … `EOF`. Có thể dùng công cụ `str_replace` của trình soạn thảo thay cho hàm này, miễn đoạn cần thay khớp đúng một lần.

## Phụ lục B — giả định chưa kiểm chứng (agent cần xác nhận khi làm)

1. Chưa chạy được `pytest`, `ruff`, `uvicorn`, `jupyter`, Docker trong môi trường lập kế hoạch. Mọi kết luận về lỗi dựa trên đọc mã và số liệu có sẵn; riêng F01 được xác nhận bằng việc nạp trực tiếp `models/best_model.joblib` (khóa `what_if` chỉ gồm `BMI, PhysActivity, Smoker, HvyAlcoholConsump, Fruits, Veggies`).
2. F03: giả định `critic.cross_check` chấp nhận `predictor_output` chỉ gồm `risk_level`, `risk_index`, `probabilities`; nếu không, bổ sung khóa còn thiếu trong test, không sửa `critic.py`.
3. F05: cấu trúc `data_validation.json["label_counts"]` đã xác minh; hãy kiểm lại khi chạy script (`ok`, `n_cols`). Mã hóa Age/Income/Education chưa đối chiếu codebook CDC (giả định A15 của plan).
4. F09: bộ phụ thuộc tối thiểu cho API phải được xác nhận bằng venv sạch (bước F09.2); nếu `import` kéo `torch`/`matplotlib` thì cần refactor import ngoài phạm vi tài liệu này.
5. Python 3.14 (máy người dùng) khác CI (3.11/3.12): nếu một thư viện chưa hỗ trợ 3.14, ưu tiên báo lại thay vì nâng/hạ phiên bản tùy ý.
