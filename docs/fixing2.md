# fixing2.md — Kế hoạch sửa lỗi đợt 2 cho `diabetes-ai-lab`

> Tài liệu giao việc cho **agent thực thi**. Mỗi mục `Fxx` tự đủ thông tin: file cần sửa, cách sửa, lệnh kiểm chứng, cú pháp commit.
> Dự án chỉ phục vụ học tập/sàng lọc nguy cơ, **không phải công cụ chẩn đoán y khoa**. Dải miễn trừ trách nhiệm trên giao diện và trường `disclaimer` trong phản hồi API phải luôn được giữ nguyên.

---

## 0. Cách dùng và quy tắc bắt buộc

### 0.1 Bối cảnh quan trọng (đọc trước)

- Repo **đã áp dụng** các mục F01–F15 của `docs/fixing.md` (có đủ commit tương ứng trong `git log`). `fixing2.md` **không** yêu cầu làm lại chúng.
- Với F01–F15: chỉ chạy lệnh **kiểm tra trạng thái** của từng mục. Nếu đã đạt thì ghi "đã đạt" và **không sửa, không commit**. Chỉ sửa khi lệnh kiểm tra cho thấy chưa đạt.
- Ngoại lệ có việc phải làm thật: **F01 (phần bổ sung còn sót)**, **F16**, **F17**.
- Đợt rà soát trước chỉ chạy được `torch`-free: 138 passed, 1 skipped. Agent phải chạy bộ test đầy đủ (có `torch`) và ghi lại số thực đo.

### 0.2 Khởi động mỗi phiên (Git Bash, Windows)

Đường dẫn có thể chứa dấu tiếng Việt/khoảng trắng nên luôn đặt trong dấu nháy:

```bash
cd "/f/PT các HTTM/AITRAINING/diabetes-ai-lab" && source .venv/Scripts/activate
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
```

Nếu repo ở chỗ khác thì `cd` tới thư mục chứa `pyproject.toml`. Python của người dùng: 3.14.5; CI dùng 3.11 và 3.12.

### 0.3 Quy tắc cứng (không được vi phạm)

1. **Không chạm vào kết quả test đã khóa:** `reports/metrics/test_lock.json`, `reports/metrics/test_evaluation.json` (không sửa, không xóa, không tạo lại). Không chạy `python scripts/evaluate.py` thiếu `--split val`. Không chạy `run_all.sh --replay-test` hoặc `--clean`.
2. **Không huấn luyện lại, không đổi seed, không đổi `configs/*.yaml`.** Mọi số đo hiện có (AUC test MLP 0,7858; XGBoost nhị phân 0,8244…) phải giữ nguyên. Không chạy lại `scripts/export_model.py` trong đợt này.
3. **Tập test chỉ được nạp bởi `scripts/evaluate.py`.** Sau mọi thay đổi, lệnh sau chỉ được trả về dòng của `scripts/evaluate.py`:
   ```bash
   grep -rn "include_test=True" src scripts
   ```
4. **Quy ước mã:** `pathlib`; mở file luôn `encoding="utf-8"`; `json.dump(..., ensure_ascii=False, indent=2)`; type hint đầy đủ; thông báo lỗi tiếng Việt nêu rõ trường/giá trị; docstring tiếng Việt; matplotlib backend `Agg`.
5. **Giữ tương thích ngược:** không đổi tên khóa JSON của `/predict`, `/explain`, `/health` (`/health` luôn trả đúng `{"status": "ok"}`); không đổi `schema_version` của bundle (đang là `1`).
6. **Kết thúc dòng (CRLF/LF) lẫn lộn.** Các file sau dùng **CRLF**: `app/static/js/app.js`, `tests/test_demo_static.py`, `src/diabetes_ai/inference/predictor.py`, `src/diabetes_ai/serving/app.py` (và có thể các file khác). Khi sửa file bằng script, dùng hàm `sub()` ở Phụ lục A (đọc/ghi bytes). Công cụ `ruff --fix`/`ruff format` giữ nguyên kiểu xuống dòng của từng file. Sau khi sửa, kiểm tra không đổi kiểu xuống dòng bằng lệnh ở Phụ lục A.
7. **Khi một bước thất bại:** dừng lại, báo lỗi kèm số liệu đo được; không tự sửa dữ liệu/tiêu chí, không dùng test để chỉnh tham số, không dùng `skip` hay `# noqa` để che lỗi.
8. **Commit:** Conventional Commits (`fix|feat|test|docs|chore|ci|style(<scope>): ...`). Mỗi mục một commit riêng. `git status --short` phải rỗng sau mỗi mục. **F17 bắt buộc là commit riêng, chỉ chứa thay đổi định dạng.**

### 0.4 Bước nền B0 (làm trước mọi mục)

```bash
git status --short                                # kỳ vọng: rỗng (nếu không, dừng và báo)
python -m pytest -q 2>&1 | tail -15               # GHI LẠI passed/failed/skipped
python -m ruff --version                          # GHI LẠI phiên bản (kỳ vọng 0.17.x)
python -m ruff check src tests --statistics 2>&1 | tail -15   # GHI LẠI số lỗi lint hiện tại
git log --oneline | head -30                      # xác nhận các commit F01–F15 đã có
```

Ghi các kết quả này vào báo cáo cuối. Nếu `git status` không rỗng, dừng lại và báo.

### 0.5 Thứ tự thực hiện

```
B0 → kiểm tra F01–F15 (chỉ đọc) → F01-bổ sung → F16 → F17 → Mục 6 (nghiệm thu)
```

F16 phải làm **trước** F17 (dọn import thừa trước khi định dạng). F17 là commit cuối cùng chạm tới mã nguồn.

---

## 1. Bảng tổng hợp

| ID | Mức | Trạng thái dự kiến | Việc phải làm |
|---|---|---|---|
| F01 | P0 | Đã áp dụng, **còn sót** | Sửa danh sách gắn sự kiện `change` trong `app.js` + thêm test `test_every_whatif_toggle_has_change_listener` |
| F02–F15 | P0–P2 | Đã áp dụng | Chỉ kiểm tra trạng thái (Mục 2) |
| F16 | P1 | **Mới** | Ghim rule ruff trong `pyproject.toml`, ghim phiên bản ruff (CI, requirements-dev, pre-commit), dọn import thừa |
| F17 | P2 | **Mới** | `ruff format` toàn bộ `src tests` trong **một commit riêng** `style: ruff format`, rồi thêm bước format-check vào CI |

---

## 2. F02–F15: kiểm tra trạng thái (không sửa nếu đã đạt)

Chạy lần lượt; mỗi dòng ghi "đạt/chưa đạt" vào báo cáo.

| ID | Nội dung | Lệnh kiểm tra | Điều kiện đạt |
|---|---|---|---|
| F02 | `/explain` trả 422 cho khóa lạ/ngoài miền/NaN | `grep -n "WhatIfError\|isfinite" src/diabetes_ai/explain/what_if.py src/diabetes_ai/serving/routes.py src/diabetes_ai/inference/schemas.py src/diabetes_ai/serving/app.py` | có `WhatIfError`, `np.isfinite`, `field_validator`, `_json_safe` |
| F03 | `SafetyAgent` lấy P(tiểu đường) theo tên lớp | `grep -n "_extract_p_diabetes\|casefold" src/diabetes_ai/agents/safety_agent.py` | có cả hai; không còn chuỗi `"Tiểu đường"` hoa |
| F04 | Test không phụ thuộc `models/best_model.joblib` | tạm `mv models/best_model.joblib /tmp/bm.bak`, chạy `python -m pytest -q tests/test_agents.py tests/test_api_predict.py tests/test_api_health.py`, rồi `mv` trả lại | pass (kỳ vọng 23 passed) |
| F05 | Notebook EDA, `eda_*.png`, từ điển dữ liệu | `ls reports/figures/eda_*.png \| wc -l` và `wc -c docs/data_dictionary.md notebooks/01_eda_brfss.ipynb` | 7 hình; hai file không còn là stub |
| F06 | `.env` có hiệu lực qua `get_env` | `grep -n "get_env" src/diabetes_ai/serving/deps.py src/diabetes_ai/inference/predictor.py` | có `get_env("MODEL_PATH")` |
| F07 | Tài liệu khớp mã | `grep -rn "24.206\|229.474\|60/20/20\|/explain/global\|BatchNorm" README.md docs/architecture.md reports/final_report.md` | không có kết quả |
| F08 | Endpoint `/consult` + nút giao diện | `grep -n "consult" src/diabetes_ai/serving/routes.py src/diabetes_ai/serving/deps.py app/static/js/app.js app/index.html` | có route, singleton, hàm `callConsult`, `consult-btn` |
| F09 | Hạ tầng | `cat -A Makefile \| head -8`; `cat Dockerfile docker-compose.yml .dockerignore requirements-serve.txt .github/workflows/ci.yml` | `all:` và dòng lệnh bắt đầu bằng TAB (`^I`); Dockerfile dùng `requirements-serve.txt`, chạy `appuser`, có `HEALTHCHECK`; compose không có `env_file` |
| F10 | Bundle ghi phiên bản thư viện | `python -c "from diabetes_ai.inference.export import load_bundle; print(load_bundle('models/best_model.joblib').get('env'))"` | in ra dict có `python`, `numpy`, `scikit-learn`; `cat reports/metrics/export_summary.json` có `"path": "models/best_model.joblib"` |
| F11 | Không dùng `innerHTML` động | `python -m pytest -q tests/test_demo_static.py -k inner_html` | passed |
| F12 | Cache cấu hình what-if; có `/ready` | `grep -n "lru_cache" src/diabetes_ai/inference/predictor.py`; `grep -n "/ready" src/diabetes_ai/serving/app.py` | có cả hai |
| F13 | `run_all.sh` kiểm khối AUTO | `grep -n "make_reports.py --check" scripts/run_all.sh` | có |
| F14 | `env_check.json` ghi phiên bản gói | `grep -n "version" scripts/check_env.py` | có ghi phiên bản gói |
| F15 | Giải thích cách đọc `label` | `grep -n -i "tỉ lệ cảnh báo\|label" reports/model_card.md reports/final_report.md` | có đoạn giải thích |

Nếu một mục **chưa đạt**: dừng lại, báo cho người dùng (không tự áp dụng lại toàn bộ `fixing.md`; hỏi trước).

---

## 3. F01 (bổ sung) — Danh sách gắn sự kiện `change` thiếu hai ô what-if mới

**Bằng chứng (đã xác minh bằng chạy thật):** sau khi F01 gốc đổi UI sang 5 ô lối sống hợp lệ (`Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`), khối `DOMContentLoaded` trong `app/static/js/app.js` vẫn gắn sự kiện `change` cho danh sách cũ `["whatif-HighBP", "whatif-HighChol", "whatif-Smoker", "whatif-PhysActivity", "whatif-Fruits"]`. Hậu quả: bật/tắt **Ăn rau** và **Uống rượu bia nhiều** không gọi `triggerWhatIfUpdate`, bảng what-if không cập nhật. (`HighBP`/`HighChol` không gây lỗi nhờ `if (el)`, nhưng là mã chết.)

**File cần sửa:** `app/static/js/app.js` (CRLF), `tests/test_demo_static.py` (CRLF).

**Kiểm tra trạng thái trước khi sửa:**

```bash
grep -n '"whatif-HighBP"' app/static/js/app.js     # nếu có kết quả → còn lỗi, tiếp tục; nếu rỗng → đã sửa, chỉ chạy kiểm chứng
```

**Bước F01.1 — `app/static/js/app.js`.** Lưu hàm `sub()` (Phụ lục A) vào `/tmp/sub.py` rồi chạy:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "/tmp")
from sub import sub

sub("app/static/js/app.js",
r'''  ["whatif-HighBP", "whatif-HighChol", "whatif-Smoker", "whatif-PhysActivity", "whatif-Fruits"].forEach((id) => {''',
r'''  [
    "whatif-Smoker",
    "whatif-PhysActivity",
    "whatif-Fruits",
    "whatif-Veggies",
    "whatif-HvyAlcoholConsump",
  ].forEach((id) => {''')
EOF
```

**Bước F01.2 — test chống tái phát.** Thêm vào `tests/test_demo_static.py`, **ngay trước** `def test_app_js_calls_consult_endpoint`:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "/tmp")
from sub import sub

sub("tests/test_demo_static.py",
r'''def test_app_js_calls_consult_endpoint() -> None:''',
r'''def test_every_whatif_toggle_has_change_listener() -> None:
    """Mọi ô bật/tắt what-if trong index.html phải được gắn sự kiện `change` trong app.js."""
    js = (ROOT / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    toggles = set(re.findall(r'<input type="checkbox" id="(whatif-\w+)"', html))
    assert toggles, "Không tìm thấy ô bật/tắt what-if nào trong index.html"
    listener = re.search(r"\[([^\]]*)\]\.forEach\(\(id\)", js)
    assert listener, "Không tìm thấy danh sách id gắn sự kiện trong app.js"
    wired = set(re.findall(r'"(whatif-\w+)"', listener.group(1)))
    assert toggles <= wired, f"Ô what-if chưa gắn sự kiện: {sorted(toggles - wired)}"
    assert wired <= toggles, f"app.js gắn sự kiện cho id không tồn tại: {sorted(wired - toggles)}"


def test_app_js_calls_consult_endpoint() -> None:''')
EOF
```

(`re` và `ROOT` đã được import sẵn ở đầu `tests/test_demo_static.py`; xác nhận bằng `grep -n "^import re\|import ROOT\|ROOT" tests/test_demo_static.py`.)

**Kiểm chứng F01:**

```bash
python -m pytest -q tests/test_demo_static.py       # kỳ vọng: 7 passed

# Chứng minh test bắt được lỗi cũ (khôi phục ngay sau đó):
git stash push app/static/js/app.js
python -m pytest -q tests/test_demo_static.py       # kỳ vọng: FAIL, thông báo: Ô what-if chưa gắn sự kiện: ['whatif-HvyAlcoholConsump', 'whatif-Veggies']
git stash pop
python -m pytest -q tests/test_demo_static.py       # kỳ vọng: 7 passed
```

(Nếu `git stash` không áp dụng được vì file vừa sửa chưa commit và test mới nằm ở file khác, vẫn đúng: stash chỉ cất `app.js`; test mới nằm lại trong working tree.)

Kiểm tra tay: `uvicorn diabetes_ai.serving.app:app --port 8200`, mở `http://localhost:8200/app/`, bấm **Đánh giá nguy cơ**, bật/tắt lần lượt **Ăn rau** và **Uống rượu bia nhiều**: bảng "Trước / Sau / Chênh lệch" phải đổi giá trị; tab Network chỉ có `/explain` mã 200.

**Kiểm tra kiểu xuống dòng không đổi:** dùng lệnh ở Phụ lục A cho `app/static/js/app.js` và `tests/test_demo_static.py` (kỳ vọng: `LF-only=0`, số CRLF tăng đúng bằng số dòng thêm).

**Commit:** `fix(demo): gắn sự kiện change cho đủ 5 ô what-if hợp lệ và thêm test chống tái phát`

---

## 4. F16 — Ghim quy tắc và phiên bản ruff, dọn import thừa

**Bằng chứng (đã đo):**
- `pyproject.toml` không có `[tool.ruff.lint]`; ruff 0.17.x dùng bộ quy tắc mặc định rộng (S, PERF, TRY, RUF…) → `ruff check src tests` báo **150 lỗi**. Với bộ cổ điển `E4,E7,E9,F` chỉ còn **31 lỗi, toàn bộ là F401 (import thừa)** và đều tự sửa được.
- `requirements-dev.txt` cài `ruff` không ghim phiên bản; `.pre-commit-config.yaml` dùng `ruff-pre-commit v0.6.9` (lệch xa bản dùng thật) → kết quả lint/format khác nhau giữa máy, pre-commit và CI.
- `Alert` được import thừa ở `agents/safety_agent.py` và `inference/predictor.py`; đã xác minh không module nào import lại `Alert` từ hai file đó (`inference/__init__.py` import từ `inference/schemas.py`).

**File cần sửa:** `pyproject.toml` (LF), `requirements-dev.txt`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, và các file `src/`, `tests/` có import thừa (do `ruff --fix` xử lý).

**Bước F16.1 — chọn phiên bản ruff để ghim.** Dùng đúng phiên bản đang dùng cục bộ:

```bash
python -m ruff --version        # ví dụ: ruff 0.17.3  → ký hiệu RUFF_VER=0.17.3 (dùng cho các bước dưới)
```

**Bước F16.2 — `pyproject.toml`.** Thêm ngay sau khối `[tool.ruff]` (sau dòng `src = ["src", "tests"]`):

```toml
[tool.ruff.lint]
# Ghim bộ quy tắc (khớp mặc định cổ điển) để CI không đổi kết quả khi ruff mở rộng bộ quy tắc mặc định.
select = ["E4", "E7", "E9", "F"]
```

**Bước F16.3 — dọn import thừa (không đổi hành vi):**

```bash
python -m ruff check src tests --fix
python -m ruff check src tests          # kỳ vọng: All checks passed!
git diff --stat                          # kỳ vọng: chỉ dòng import bị xóa/rút gọn, không đổi logic
python -m pytest -q                      # kỳ vọng: số passed không giảm so với B0
```

Không dùng `--unsafe-fixes`. Nếu `ruff` đòi sửa import mà việc xóa làm test fail, **dừng và báo**, không thêm `# noqa`.

**Bước F16.4 — ghim phiên bản ruff.**
(a) `requirements-dev.txt`: đổi dòng `ruff` thành `ruff==<RUFF_VER>`.
(b) `.pre-commit-config.yaml`: đổi `rev: v0.6.9` của `ruff-pre-commit` thành `rev: v<RUFF_VER>` (giữ nguyên `hooks` và `args`).
(c) `.github/workflows/ci.yml`: thêm bước kiểm tra phiên bản ngay trước bước `ruff check`, để CI hỏng rõ ràng nếu ai đổi pin ở một nơi mà quên nơi khác:

```yaml
      - name: Kiểm tra phiên bản ruff đã ghim
        run: ruff --version | grep -q "<RUFF_VER>"
```

(thay `<RUFF_VER>` bằng số thật, ví dụ `0.17.3`). Giữ nguyên các bước còn lại.

**Kiểm chứng F16:**

```bash
pip install -r requirements-dev.txt        # nếu cần, để môi trường khớp bản ghim
python -m ruff --version                   # khớp RUFF_VER
python -m ruff check src tests             # All checks passed!
python -m pytest -q                        # số passed không giảm so với B0
git status --short                         # chỉ các file thuộc F16
```

Nếu có Python ≥ 3.11 sạch, nên thử `act`/hoặc ít nhất đọc lại `ci.yml` để chắc cú pháp YAML đúng (`python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml',encoding='utf-8'))"`).

**Commit:** `chore(lint): ghim quy tắc và phiên bản ruff, dọn import thừa`

---

## 5. F17 — Định dạng toàn bộ mã bằng `ruff format` (commit riêng)

**Bằng chứng (đã đo):** `ruff format --check src tests` báo **46 file cần định dạng lại**, nên `make lint` (có `ruff format --check`) fail. Thay đổi này chỉ là định dạng, phải **tách thành commit riêng** để không làm nhiễu lịch sử Git (dễ `git blame`, dễ revert).

**Điều kiện tiên quyết:** F16 đã commit, `git status --short` rỗng.

**Bước F17.1 — định dạng:**

```bash
python -m ruff format --check src tests 2>&1 | tail -5     # GHI LẠI số file cần định dạng (kỳ vọng ~46)
python -m ruff format src tests
python -m ruff format --check src tests                    # kỳ vọng: all files already formatted
git diff --stat | tail -5                                  # chỉ file .py trong src/ và tests/
```

**Bước F17.2 — chứng minh chỉ đổi định dạng, không đổi logic:**

```bash
python -m ruff check src tests                             # All checks passed!
python -m pytest -q                                        # số passed bằng đúng B0/F16
grep -rn "include_test=True" src scripts                   # vẫn chỉ scripts/evaluate.py
git diff --stat -- reports/metrics/test_lock.json reports/metrics/test_evaluation.json   # kỳ vọng: rỗng
```

Kiểm tra không có file ngoài `src/` và `tests/` bị đổi: `git diff --name-only | grep -v '^src/\|^tests/'` phải rỗng. Nếu `ruff format` làm đổi kiểu xuống dòng của file (CRLF ↔ LF), dừng và báo (`ruff` mặc định giữ nguyên, không nên xảy ra).

**Commit riêng (chỉ định dạng):**

```bash
git add -A src tests
git commit -m "style: ruff format"
git status --short        # rỗng
```

**Bước F17.3 — thêm kiểm tra định dạng vào CI (commit riêng, sau commit style).** Trong `.github/workflows/ci.yml`, thêm ngay sau `- run: ruff check src tests`:

```yaml
      - run: ruff format --check src tests
```

Kiểm chứng: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml',encoding='utf-8'))"`, rồi `make lint` (nếu có `make`) hoặc `python -m ruff check src tests && python -m ruff format --check src tests` phải thoát với mã 0.

**Commit:** `ci: kiểm tra ruff format trong CI`

---

## 6. Nghiệm thu cuối (làm sau khi xong mọi mục)

```bash
# 1. Toàn bộ test và lint (so với B0)
python -m pytest -q 2>&1 | tail -15
python -m ruff check src tests
python -m ruff format --check src tests

# 2. Bản clone sạch (mô phỏng CI)
rm -rf /tmp/dal-clean && git clone . /tmp/dal-clean && (cd /tmp/dal-clean && python -m pytest -q 2>&1 | tail -15)

# 3. Kết quả đã khóa không bị đụng tới (N = số commit của đợt này)
git diff --stat HEAD~N -- reports/metrics/test_evaluation.json reports/metrics/test_lock.json   # kỳ vọng: rỗng
grep -rn "include_test=True" src scripts      # kỳ vọng: chỉ scripts/evaluate.py

# 4. Báo cáo tự sinh vẫn đầy đủ
python scripts/make_reports.py --check; echo $?     # 0

# 5. Smoke test thật (tự khởi động và tắt máy chủ)
python - <<'EOF'
import json, subprocess, sys, time, httpx
base = "http://127.0.0.1:8200"
p = subprocess.Popen([sys.executable, "-m", "uvicorn", "diabetes_ai.serving.app:app", "--port", "8200"])
try:
    for _ in range(60):
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

**Checklist hoàn tất** (dán vào báo cáo cuối, kèm số đo thực tế):

| Mục | Điều kiện đạt |
|---|---|
| F01 | `test_every_whatif_toggle_has_change_listener` pass; test fail khi hoàn nguyên `app.js`; bật/tắt 5 ô what-if đều cập nhật bảng |
| F02–F15 | mỗi mục "đạt" theo bảng Mục 2 (hoặc nêu rõ lý do chưa đạt) |
| F16 | `select` đã ghim; `ruff==<ver>` ở `requirements-dev.txt`, `.pre-commit-config.yaml`, `ci.yml`; `ruff check` sạch; pytest không giảm |
| F17 | commit `style: ruff format` chỉ chứa thay đổi định dạng; `ruff format --check` sạch; CI có bước format-check |
| Chung | test đã khóa nguyên vẹn; `include_test=True` chỉ ở `evaluate.py`; `make_reports.py --check` = 0; clone sạch pass |

**Nếu một mục không làm được** (ví dụ không có Docker, thiếu `torch`), ghi rõ lý do và phần chưa kiểm chứng; không đánh dấu hoàn tất.

---

## Phụ lục A — hàm `sub()` để sửa file giữ nguyên CRLF

Lưu thành `/tmp/sub.py` (công cụ tạm, **không commit**):

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

Kiểm tra kiểu xuống dòng của một file (dùng Python, **không** dùng `grep $'\r'` vì không chạy trong mọi shell):

```bash
python - <<'EOF'
from pathlib import Path
for f in ["app/static/js/app.js", "tests/test_demo_static.py"]:
    b = Path(f).read_bytes(); crlf = b.count(b"\r\n")
    print(f, "CRLF =", crlf, "| LF-only =", b.count(b"\n") - crlf)
EOF
```

## Phụ lục B — giả định chưa kiểm chứng

1. Phần F02–F15 chỉ được xác minh bằng đọc mã/chạy test ở phạm vi F01–F04, F08–F10; F05–F07, F11–F15 chưa được kiểm tra trực tiếp ở đợt rà soát trước, nên Mục 2 yêu cầu agent tự xác nhận.
2. Chưa chạy Docker; chưa chạy 3 file test cần `torch` (`test_mlp.py`, `test_trainer.py`, `test_probabilities_sum_to_one.py`) và 1 test trong `test_predictor.py` ở đợt rà soát trước.
3. Python 3.14 (máy người dùng) khác CI (3.11/3.12). Nếu một thư viện chưa hỗ trợ 3.14, báo lại thay vì nâng/hạ phiên bản tùy ý.
4. Phiên bản ruff được ghim theo bản cài trên máy người dùng (kỳ vọng 0.17.x). Kết quả `ruff format` có thể khác nhẹ giữa các phiên bản nhỏ; vì vậy phải ghim đồng nhất ở `requirements-dev.txt`, `.pre-commit-config.yaml` và `ci.yml`.