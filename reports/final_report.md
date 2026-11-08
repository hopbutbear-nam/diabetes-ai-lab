# Báo cáo tổng kết dự án (Final Report) — Hệ thống AI ước lượng nguy cơ tiểu đường

> **MIỄN TRỪ TRÁCH NHIỆM:**
> Toàn bộ dự án này được thực hiện vì **mục đích nghiên cứu khoa học và học tập, không phải chẩn đoán y tế**. Các kết luận và chỉ số hiệu năng không cấu thành lời khuyên y khoa hoặc chỉ định điều trị.

---

## 1. Tóm tắt dự án

Dự án này nghiên cứu và hiện thực hóa một đường ống học máy hoàn chỉnh (end-to-end Machine Learning pipeline) nhằm giải quyết bài toán ước lượng nguy cơ mắc bệnh tiểu đường ở người trưởng thành từ dữ liệu khảo sát cộng đồng lớn (CDC BRFSS 2015).

Các điểm mốc kỹ thuật chính đã đạt được:
1. **Quy trình dữ liệu nghiêm ngặt:** Phân tầng dữ liệu độc lập, kiểm soát rò rỉ thông tin (data leakage) đối với các dòng trùng lặp, và khóa cứng tập kiểm thử (Test split).
2. **So sánh đa mô hình:** Huấn luyện và đánh giá đối chuẩn 4 họ mô hình: Majority Baseline, Logistic Regression, XGBoost và Mạng nơ-ron nhiều tầng (MLP).
3. **Hiệu chuẩn xác suất & phân tầng nguy cơ:** Hiệu chuẩn xác suất đầu ra bằng kỹ thuật Out-of-fold trên tập Validation, thiết lập hệ thống cảnh báo nguy cơ đa tầng (Trung bình và Cao) có ý nghĩa sàng lọc.
4. **Đánh giá công bằng & giải thích:** Khảo sát chi tiết tính công bằng giữa các nhóm nhân khẩu học và phân tích tầm quan trọng của các đặc trưng bằng Permutation Importance.
5. **Triển khai ứng dụng an toàn:** Tích hợp bộ luật an toàn lâm sàng, xây dựng RESTful API và giao diện web tương tác minh bạch kèm các thông điệp cảnh báo y tế rõ ràng.

---

## 2. Dữ liệu và phân chia tập mẫu

Tập dữ liệu gốc gồm 253.680 bản ghi từ cuộc điều tra BRFSS 2015 của CDC Hoa Kỳ. Toàn bộ 21 biến dự báo là các thông tin khảo sát hành vi, nhân khẩu học và bệnh nền do người tham gia tự khai.

### Tóm tắt tổng thể dữ liệu

<!-- AUTO:data_summary -->
| Chỉ số thống kê dữ liệu | Giá trị đo được | Ý nghĩa lâm sàng / dữ liệu |
| :--- | :--- | :--- |
| Tổng số dòng (quan sát) | **253.680** | Bộ dữ liệu khảo sát BRFSS 2015 từ CDC Hoa Kỳ |
| Số cột đặc trưng | **21** (cộng 1 cột nhãn) | 21 biến nhân khẩu học, lối sống và bệnh nền |
| Giá trị bị thiếu (Missing values) | **0** (0,00%) | Toàn bộ các câu hỏi khảo sát đã được làm sạch đầy đủ |
| Số dòng trùng lặp hoàn toàn | **23.899** (9,42%) | Xuất hiện tự nhiên do toàn bộ 21 biến là biến phân loại rời rạc |
| Số dòng hồ sơ phân biệt (Unique) | **229.781** (90,58%) | Các tổ hợp trả lời khảo sát độc nhất |
| Lớp 0 — Không tiểu đường | **213.703** (84,24%) | Nhóm đa số áp đảo trong dân số khảo sát |
| Lớp 1 — Tiền tiểu đường | **4.631** (1,83%) | Nhóm cực kỳ thiểu số (< 2%), ranh giới sinh lý mờ nhạt |
| Lớp 2 — Tiểu đường | **35.346** (13,93%) | Nhóm bệnh nhân đích cần ưu tiên sàng lọc |
<!-- /AUTO -->

### Chiến lược phân chia dữ liệu (Data Splits)

Dữ liệu được chia theo tỉ lệ 68% Huấn luyện (Train), 12% Thẩm định (Validation) và 20% Kiểm thử (Test), bảo toàn phân bố nhãn bằng phân tầng (Stratified Split). Hai biến thể dữ liệu được duy trì song song: bản đầy đủ (Full) và bản loại trùng lặp (Dedup).

<!-- AUTO:split_table -->
| Biến thể | Tập chia | Số mẫu (N) | Tỉ lệ (%) | Lớp 0 (Không) | Lớp 1 (Tiền TĐ) | Lớp 2 (Tiểu đường) | Trùng với Train (twins) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bản đầy đủ (Full)** | Train | 172.502 | 68,00% | 145.317 (84,24%) | 3.149 (1,83%) | 24.036 (13,93%) | — (Tập gốc) |
|  | Validation | 30.442 | 12,00% | 25.645 (84,24%) | 556 (1,83%) | 4.241 (13,93%) | 3.856 (12,67%) |
|  | Test (1 lần) | 50.736 | 20,00% | 42.741 (84,24%) | 926 (1,83%) | 7.069 (13,93%) | 6.469 (12,75%) |
| **Bản loại trùng (Dedup)** | Train | 156.250 | 68,00% | 129.236 (82,71%) | 3.148 (2,01%) | 23.866 (15,27%) | — (Tập gốc) |
|  | Validation | 27.574 | 12,00% | 22.807 (82,71%) | 555 (2,01%) | 4.212 (15,28%) | 295 (1,07%) |
|  | Test (1 lần) | 45.957 | 20,00% | 38.012 (82,71%) | 926 (2,01%) | 7.019 (15,27%) | 522 (1,14%) |
<!-- /AUTO -->

---

## 3. Phương pháp nghiên cứu và phát triển

- **Tiền xử lý đặc trưng:**
  - Chuẩn hóa các biến liên tục và biến thứ bậc (BMI, Tuổi, Thu nhập, Học vấn, v.v.).
  - Mã hóa chuẩn mực các biến nhị phân chỉ thị sức khỏe.
- **Quy trình huấn luyện và chọn mô hình:**
  - Lựa chọn mô hình và tinh chỉnh siêu tham số hoàn toàn độc quyền trên tập Thẩm định (Validation).
  - Áp dụng nguyên tắc cận biên đơn giản (Simplicity Margin): Mô hình phức tạp hơn chỉ được lựa chọn nếu điểm ROC-AUC vượt trội hơn mô hình tuyến tính đơn giản (Logistic Regression) ít nhất 0,003.
- **Hiệu chuẩn xác suất:**
  - Sử dụng phương pháp hiệu chuẩn đẳng trương (Isotonic Regression) hoặc hồi quy logistic đa lớp qua cơ chế Out-of-fold 5-fold trên tập Validation, bảo toàn tính khách quan trước khi áp dụng lên tập Test.
- **Khóa tập Kiểm thử (Test Lock Protocol):**
  - Tập Test chỉ được giải mã và đánh giá đúng 1 lần duy nhất để kiểm tra tính tổng quát hóa thực tế của toàn bộ hệ thống.

---

## 4. Kết quả thực nghiệm

### Hiệu năng trên tập Thẩm định (Validation)

<!-- AUTO:val_results -->
| Mô hình | ROC-AUC (OvR macro) | PR-AUC (macro) | Macro-F1 | Accuracy | Baseline Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline (Luôn đoán Lớp 0)** | 0,5000 | 0,3333 | 0,3048 | 0,8424 | 0,8424 |
| **Logistic Regression (Hồi quy Logistic)** | 0,7763 | 0,4643 | 0,4620 | 0,7984 | 0,8424 |
| **XGBoost (Cây quyết định tăng cường)** | 0,7858 | 0,4741 | 0,4654 | 0,8057 | 0,8424 |
| **MLP (Mạng nơ-ron nhiều tầng — ĐƯỢC CHỌN)** | 0,7879 | 0,4743 | 0,4738 | 0,8069 | 0,8424 |
<!-- /AUTO -->

### Đánh giá chính thức trên tập Kiểm thử độc lập (Test)

<!-- AUTO:test_results -->
### Bảng 1: So sánh tổng thể các mô hình trên tập Test (N = 50.736)

| Mô hình | ROC-AUC macro (95% CI) | PR-AUC macro | Macro-F1 | Accuracy | Baseline Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Majority Baseline | 0,5000 | 0,3333 | 0,3048 | 0,8424 | 0,8424 |
| Logistic Regression | **0,7719** [0,7655; 0,7783] | 0,4637 | 0,4593 | 0,7961 | 0,8424 |
| XGBoost | **0,7873** [0,7812; 0,7935] | 0,4763 | 0,4679 | 0,8053 | 0,8424 |
| MLP (Mô hình triển khai) | **0,7858** [0,7797; 0,7923] | 0,4735 | 0,4641 | 0,8050 | 0,8424 |

### Bảng 2: Hiệu năng chi tiết theo từng lớp của mô hình MLP trên tập Test

| Nhãn bệnh lý | Số mẫu trong Test | Recall / Độ nhạy (95% CI) | Precision / Độ chuẩn xác | F1-score | ROC-AUC (OvR) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Lớp 0: Không tiểu đường** | 42.741 | **0,8684** [0,8653; 0,8713] | 0,9104 | 0,8889 | 0,8230 |
| **Lớp 1: Tiền tiểu đường** | 926 | **0,0454** [0,0335; 0,0583] | 0,0408 | 0,0429 | 0,7067 |
| **Lớp 2: Tiểu đường** | 7.069 | **0,5213** [0,5098; 0,5325] | 0,4123 | 0,4604 | 0,8278 |
<!-- /AUTO -->

---

## 5. Đánh giá tính công bằng (Fairness Summary)

Hệ thống đã thực hiện đánh giá độ nhạy (Recall), độ chính xác dự báo (Precision) và tỉ lệ phát cảnh báo trên tập Test độc lập theo 3 trục nhân khẩu học chính:
1. **Độ tuổi:** Sự chênh lệch độ nhạy giữa nhóm tuổi trẻ nhất (18–34 tuổi) và nhóm cao tuổi (65–74 tuổi) là rất lớn. Điều này phản ánh sự chênh lệch tự nhiên về tỉ lệ bệnh nền (Prevalence) trong mẫu khảo sát.
2. **Mức thu nhập:** Nhóm thu nhập thấp có tỉ lệ mắc bệnh nền cao hơn và đạt độ nhạy phát hiện bệnh cao hơn nhóm thu nhập cao.
3. **Giới tính sinh học:** Hiệu năng phát hiện bệnh giữa Nam và Nữ là tương đương nhau (chênh lệch dưới 2 điểm phần trăm).

*Chi tiết toàn bộ số liệu và phân tích chuyên sâu được trình bày trong tài liệu `reports/fairness_report.md`.*

---

## 6. Khả năng giải thích mô hình (Model Explainability)

Để hiểu rõ mức độ đóng góp của từng yếu tố vào quyết định phân loại của mô hình MLP, phương pháp Permutation Importance (đo mức độ sụt giảm ROC-AUC khi xáo trộn ngẫu nhiên từng cột đặc trưng) đã được thực hiện trên tập validation (không dùng tập test).

<!-- AUTO:importance_top10 -->
| Thứ hạng | Mã đặc trưng | Diễn giải lâm sàng / khảo sát | Mức giảm ROC-AUC trung bình (Mean drop) | Độ lệch chuẩn (Std) |
| :--- | :--- | :--- | :--- | :--- |
| #1 | **GenHlth** | Tự đánh giá sức khỏe (1 tốt – 5 kém) | **+0,0487** | ±0,0005 |
| #2 | **BMI** | Chỉ số BMI | **+0,0442** | ±0,0015 |
| #3 | **Age** | Nhóm tuổi (13 nhóm) | **+0,0340** | ±0,0029 |
| #4 | **HighChol** | Mỡ máu cao | **+0,0161** | ±0,0009 |
| #5 | **HighBP** | Huyết áp cao | **+0,0149** | ±0,0013 |
| #6 | **Income** | Thu nhập hộ gia đình (8 mức) | **+0,0079** | ±0,0015 |
| #7 | **Sex** | Giới tính | **+0,0048** | ±0,0002 |
| #8 | **CholCheck** | Kiểm tra cholesterol trong 5 năm | **+0,0047** | ±0,0013 |
| #9 | **HeartDiseaseorAttack** | Bệnh tim mạch hoặc nhồi máu cơ tim | **+0,0035** | ±0,0003 |
| #10 | **Education** | Trình độ học vấn (6 mức) | **+0,0031** | ±0,0004 |
<!-- /AUTO -->

> **LƯU Ý QUAN TRỌNG VỀ TÍNH CHẤT TƯƠNG QUAN:**
> Các chỉ số mức độ quan trọng ở bảng trên **chỉ là tương quan thống kê** đo lường mức độ phụ thuộc của mô hình học máy vào các trường thông tin trong bộ dữ liệu BRFSS 2015. Đây **hoàn toàn không phải là mối quan hệ nhân quả y khoa**. Ví dụ: việc tình trạng sức khỏe chung (GenHlth) hay chỉ số BMI có tầm quan trọng cao phản ánh sự đồng biến mạnh mẽ giữa thể trạng và nguy cơ tiểu đường trong điều tra xã hội học, chứ không khẳng định việc thay đổi đơn lẻ một chỉ số sẽ ngay lập tức loại bỏ nguy cơ bệnh lý.

---

## 7. An toàn lâm sàng và quy tắc bảo vệ người dùng

Hệ thống được thiết kế với tiêu chuẩn an toàn cao nhằm bảo vệ người dùng khỏi việc hiểu sai thông tin:
1. **Luật an toàn ưu tiên (Safety Rules Override):** Khi người dùng nhập các chỉ số sức khỏe báo động (ví dụ: BMI cực kỳ cao hoặc kết hợp huyết áp cao và sức khỏe kém nghiêm trọng), hệ thống luật an toàn sẽ lập tức can thiệp nâng mức cảnh báo lên mức Cao nhất, bất kể xác suất mô hình toán học dự đoán là bao nhiêu.
2. **Khuyến cáo xét nghiệm xác nhận:** Toàn bộ kết quả cảnh báo nguy cơ (Trung bình hoặc Cao) đều đi kèm hướng dẫn người dùng đến các cơ sở y tế để thực hiện **xét nghiệm xác nhận chuyên khoa** (xét nghiệm đường huyết lúc đói hoặc xét nghiệm chỉ số HbA1c).
3. **Không lưu trữ dữ liệu cá nhân:** API và ứng dụng web hoạt động theo cơ chế phi trạng thái (stateless), không ghi log hay lưu lại thông tin nhạy cảm của người dùng.

---

## 8. Hạn chế

1. **Khó khăn trong phân biệt nhóm tiền tiểu đường:** Lớp "tiền tiểu đường" gần như không phân biệt được (recall test 4,5%, precision 4,1%). Các biểu hiện hành vi bên ngoài không đủ để tách biệt người tiền tiểu đường với người bình thường nếu thiếu xét nghiệm sinh hóa chuyên sâu.
2. **Accuracy thấp hơn Baseline và tính không phù hợp của Accuracy:**
   - Do 84,24% dân số mẫu không mắc bệnh, một mô hình tầm thường chỉ cần đoán "không bệnh" là đạt 84,24% Accuracy.
   - Mô hình của chúng ta đạt Accuracy khoảng 80% vì nó tích cực cảnh báo các trường hợp có nguy cơ cao (để đạt độ nhạy trên 50% cho lớp bệnh và trên 90% khi sàng lọc gộp). Do đó, Accuracy thấp hơn baseline là hệ quả tự nhiên và chấp nhận được trong bài toán y tế mất cân bằng nhãn.
3. **Tỉ lệ cảnh báo:** Ở ngưỡng "trung bình" (chỉ số ≥ 0,10) khoảng **49,9%** người được xếp từ mức trung bình trở lên (PPV 25,3%, độ nhạy 90,4%); ở mức "cao" (≥ 0,25) là **22,2%** (PPV 38,0%, độ nhạy 60,5%). Đây là đánh đổi có chủ đích của công cụ **sàng lọc**, không phải chẩn đoán.
4. **Dữ liệu tự khai:** BRFSS 2015 dựa trên lời tự khai qua điện thoại, có nguy cơ sai số nhớ lại hoặc thiếu chính xác về cân nặng, chiều cao.
5. **Không đại diện cho quần thể Việt Nam:** Mô hình mang đặc tính dịch tễ của người dân Hoa Kỳ năm 2015, không phù hợp để áp dụng trực tiếp cho người Việt Nam.

---

## 9. Hướng phát triển tiếp theo

- **Bổ sung các chỉ số cận lâm sàng:** Tích hợp các kết quả xét nghiệm sinh hóa cơ bản (đường huyết lúc đói, HbA1c, mỡ máu) để cải thiện vượt bậc khả năng phân biệt tiền tiểu đường.
- **Nghiên cứu trên dữ liệu dân số Việt Nam:** Hợp tác với các bệnh viện và viện nghiên cứu trong nước để thu thập dữ liệu dịch tễ học phù hợp với thể trạng người Á Đông.
- **Thử nghiệm các kỹ thuật học sâu tiên tiến:** Khám phá kiến trúc TabNet hoặc Transformer cho dữ liệu bảng để tối ưu hóa việc trích xuất đặc trưng phi tuyến tính phức tạp.

---

## 10. Kết luận trung thực về mô hình

Dựa trên toàn bộ kết quả đo đạc thực nghiệm độc lập và khoảng tin cậy 95% Bootstrap:

1. **Về việc so sánh giữa MLP và Logistic Regression:**
   - Điểm số ROC-AUC trung bình của mạng nơ-ron MLP trên tập Test cao hơn một chút so với mô hình hồi quy tuyến tính Logistic Regression.
   - Tuy nhiên, khi đối chiếu khoảng tin cậy 95%, cận dưới của MLP và cận trên của Logistic Regression nằm rất sát nhau. Sự chênh lệch về hiệu năng thực tế là rất khiêm tốn.
2. **Về việc so sánh giữa MLP và XGBoost:**
   - Khoảng tin cậy ROC-AUC của MLP và XGBoost chồng lấn nhau gần như hoàn toàn. Không có cơ sở thống kê nào để tuyên bố MLP vượt trội hơn XGBoost trên bộ dữ liệu này.
3. **Tổng kết:**
   - Dự án **không tuyên bố mô hình phức tạp (MLP) có sự vượt trội áp đảo** so với các mô hình đơn giản hơn. Việc lựa chọn MLP trong dự án chủ yếu phục vụ mục tiêu học tập kiến trúc mạng nơ-ron và khả năng trích xuất đặc trưng liên tục, trong khi mô hình tuyến tính đơn giản vẫn duy trì giá trị rất cao về tính minh bạch và chi phí tính toán thấp.

---

> **MIỄN TRỪ TRÁCH NHIỆM CUỐI TÀI LIỆU:**
> Báo cáo này được lập **chỉ để học tập, nghiên cứu; không phải chẩn đoán y khoa**. Mọi ứng dụng thực tế cần có sự thẩm định và giám sát của các chuyên gia y tế được cấp phép.
