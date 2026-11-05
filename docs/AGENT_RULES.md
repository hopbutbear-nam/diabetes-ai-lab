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