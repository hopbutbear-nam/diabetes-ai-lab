# Từ điển dữ liệu — BRFSS 2015 (Diabetes Health Indicators)

> Tệp này do `scripts/make_data_dictionary.py` sinh từ số liệu đo; không sửa tay các con số.

## 1. Nguồn

- Tệp: `diabetes_012_health_indicators_BRFSS2015.csv` (bộ "Diabetes Health Indicators" từ khảo sát BRFSS 2015 của CDC, bản đăng trên Kaggle).
- Ngày tệp (theo `stat`): 2026-11-05; kích thước 22.738.151 byte; sha256 `a971809d0786d2d7d7f6070f83dfe8af9d860ea33a79e0e2701c8a49f78f9861`.
- Số dòng: 253.680; số cột: 22 (21 đặc trưng + 1 nhãn).

## 2. Các cột

| Cột | Tiếng Việt | Kiểu | Miền | Mã hóa |
|---|---|---|---|---|
| `HighBP` | Huyết áp cao | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `HighChol` | Mỡ máu cao | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `CholCheck` | Kiểm tra cholesterol trong 5 năm | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `BMI` | Chỉ số BMI | liên tục | 10 – 100 | 10 – 100 |
| `Smoker` | Hút thuốc (≥100 điếu) | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `Stroke` | Từng bị đột quỵ | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `HeartDiseaseorAttack` | Bệnh tim mạch hoặc nhồi máu cơ tim | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `PhysActivity` | Hoạt động thể chất (30 ngày qua) | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `Fruits` | Ăn trái cây hàng ngày | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `Veggies` | Ăn rau hàng ngày | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `HvyAlcoholConsump` | Uống nhiều rượu bia | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `AnyHealthcare` | Có bảo hiểm y tế | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `NoDocbcCost` | Không thể khám bệnh vì chi phí | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `GenHlth` | Tự đánh giá sức khỏe (1 tốt – 5 kém) | thứ bậc | 1 – 5 | 1 = Rất tốt; 2 = Tốt; 3 = Khá; 4 = Trung bình; 5 = Kém |
| `MentHlth` | Số ngày sức khỏe tinh thần không tốt (30 ngày qua) | đếm (số ngày) | 0 – 30 | 0 – 30 |
| `PhysHlth` | Số ngày sức khỏe thể chất không tốt (30 ngày qua) | đếm (số ngày) | 0 – 30 | 0 – 30 |
| `DiffWalk` | Khó khăn khi đi bộ hoặc leo cầu thang | nhị phân | 0 – 1 | 0 = không, 1 = có |
| `Sex` | Giới tính | nhị phân | 0 – 1 | 0 = Nữ; 1 = Nam |
| `Age` | Nhóm tuổi (13 nhóm) | thứ bậc | 1 – 13 | 1 = 18–24; 2 = 25–29; 3 = 30–34; 4 = 35–39; 5 = 40–44; 6 = 45–49; 7 = 50–54; 8 = 55–59; 9 = 60–64; 10 = 65–69; 11 = 70–74; 12 = 75–79; 13 = 80+ |
| `Education` | Trình độ học vấn (6 mức) | thứ bậc | 1 – 6 | 1 = Chưa từng đi học hoặc mẫu giáo; 2 = Tiểu học (lớp 1–8); 3 = THCS hoặc chưa tốt nghiệp THPT (lớp 9–11); 4 = Tốt nghiệp THPT hoặc tương đương; 5 = Cao đẳng hoặc đại học dang dở (1–3 năm); 6 = Tốt nghiệp đại học trở lên (≥4 năm) |
| `Income` | Thu nhập hộ gia đình (8 mức) | thứ bậc | 1 – 8 | 1 = <10k USD; 2 = 10–15k; 3 = 15–20k; 4 = 20–25k; 5 = 25–35k; 6 = 35–50k; 7 = 50–75k; 8 = ≥75k |

**Lưu ý (giả định A15):** cách mã hóa Age/Income/Education/GenHlth/Sex theo mô tả của bộ dữ liệu Kaggle; **chưa đối chiếu** với codebook CDC gốc.

## 3. Nhãn

- Nhãn 3 lớp `Diabetes_012`: 0 = không; 1 = tiền tiểu đường; 2 = tiểu đường.
- Nhãn nhị phân: 0 = không; 1 = tiền tiểu đường hoặc tiểu đường.
- Số lượng toàn bộ: không = 213.703; tiền tiểu đường = 4.631; tiểu đường = 35.346.

## 4. Chất lượng dữ liệu

- Dòng trùng hoàn toàn (cả nhãn): 23.899 (9,4%); còn 229.781 dòng duy nhất.
- Kết luận kiểm tra tự động: `ok = True`.

## 5. Quyết định về dòng trùng lặp

- Biến thể **`full`** (giữ dòng trùng) là biến thể **chính**: các dòng trùng là những người trả lời khác nhau có hồ sơ giống nhau; khử trùng làm lệch tỉ lệ tiểu đường nền từ 13,9% sang 15,3% và làm xác suất hiệu chuẩn sai.
- Biến thể **`dedup`** (xóa bản sao thừa, giữ lần xuất hiện đầu tiên) chỉ là **đối chứng**.

## 6. Chia tập (seed 42, phân tầng theo nhãn 3 lớp)

| Biến thể | Train | Validation | Test |
|---|---|---|---|
| full | 172.502 | 30.442 | 50.736 |
| dedup | 156.250 | 27.574 | 45.957 |

`val_size = 0,15` là 15% của phần còn lại sau khi tách test (`test_size = 0,2`).
