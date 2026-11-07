# Thẻ mô hình (Model Card) — Hệ thống ước lượng nguy cơ tiểu đường

> **MIỄN TRỪ TRÁCH NHIỆM QUAN TRỌNG:**
> Hệ thống này được xây dựng **chỉ để học tập, nghiên cứu và thử nghiệm công nghệ; không phải chẩn đoán y khoa**. Kết quả do mô hình cung cấp không thể thay thế cho ý kiến chuyên môn của bác sĩ hoặc các xét nghiệm y tế xác nhận chính thức.

---

## Mục đích và phạm vi

Mô hình này được phát triển nhằm mục đích thử nghiệm và học tập về quy trình xây dựng hệ thống trí tuệ nhân tạo y tế có trách nhiệm (Responsible Medical AI), từ khâu xử lý dữ liệu khảo sát cộng đồng, huấn luyện đa mô hình, hiệu chuẩn xác suất, phân tích độ công bằng cho tới tích hợp luật an toàn lâm sàng.

- **Nhiệm vụ chính:** Dự đoán nguy cơ mắc bệnh tiểu đường ở người trưởng thành dựa trên các đặc trưng nhân khẩu học, tiền sử bệnh lý và thói quen sinh hoạt.
- **Dạng bài toán:** Phân loại 3 lớp (Không tiểu đường, Tiền tiểu đường, Tiểu đường) và bài toán phụ phân loại nhị phân (Không mắc vs Có mắc/tiền tiểu đường).
- **Phạm vi áp dụng:** Chỉ dùng cho mục đích trình diễn công nghệ, học tập khoa học dữ liệu và nghiên cứu kỹ thuật. Mô hình **không** được cấp phép và **không** được thiết kế để sử dụng trong môi trường khám chữa bệnh thực tế.

---

## Dữ liệu

- **Nguồn dữ liệu:** Bộ dữ liệu khảo sát các yếu tố nguy cơ hành vi (BRFSS - Behavioral Risk Factor Surveillance System) năm 2015 do Trung tâm Kiểm soát và Phòng ngừa Dịch bệnh (CDC) Hoa Kỳ thu thập.
- **Phương thức thu thập:** Phỏng vấn qua điện thoại đối với người trưởng thành tại Hoa Kỳ; toàn bộ thông tin là dữ liệu **tự khai** (self-reported), không qua kiểm chứng hồ sơ bệnh án hay xét nghiệm máu tại thời điểm khảo sát.
- **Đặc điểm phân bố:**
  - Tập dữ liệu gồm các đặc trưng phân loại và thứ bậc về nhân khẩu học, hành vi sức khỏe, khả năng tiếp cận y tế và bệnh lý nền.
  - Tỉ lệ phân bố nhãn bị mất cân bằng nghiêm trọng: nhóm không mắc bệnh chiếm đa số áp đảo, nhóm tiểu đường chiếm tỉ lệ trung bình, trong khi nhóm tiền tiểu đường là lớp cực kỳ thiểu số.
  - Sự tồn tại của các dòng có giá trị đặc trưng hoàn toàn trùng lặp là hiện tượng tự nhiên do bản chất các biến khảo sát đều là biến rời rạc.

---

## Mô hình và huấn luyện

- **Kiến trúc mô hình chính:** Mạng nơ-ron nhiều tầng (MLP — Multi-Layer Perceptron) với các tầng Fully Connected, kích hoạt ReLU, chuẩn hóa BatchNorm và Dropout chống quá khớp.
- **Các mô hình đối chứng cơ sở:**
  - Majority Baseline (luôn dự đoán lớp đa số).
  - Logistic Regression (mô hình tuyến tính cơ sở).
  - XGBoost (mô hình cây quyết định tăng cường gradient).
- **Chiến lược phân chia dữ liệu:** Phân tầng theo nhãn bệnh (Stratified Split) thành 3 tập riêng biệt: Huấn luyện (Train), Thẩm định (Validation), và Kiểm thử độc lập (Test).
- **Quy tắc ngăn ngừa rò rỉ dữ liệu:** Tập Kiểm thử độc lập (Test) bị khóa hoàn toàn và chỉ được đánh giá một lần duy nhất sau khi đã chốt mô hình, trọng số, phương pháp hiệu chuẩn và ngưỡng quyết định.

---

## Hiệu năng

Mọi con số kết quả đánh giá đều được đo đạc tự động từ các kịch bản thực nghiệm, hoàn toàn không sao chép thủ công.

### Hiệu năng trên tập Thẩm định (Validation)

Quá trình lựa chọn mô hình triển khai được thực hiện độc quyền trên tập Thẩm định dựa trên chỉ số ROC-AUC (OvR macro) và PR-AUC (macro):

<!-- AUTO:val_results -->
| Mô hình | ROC-AUC (OvR macro) | PR-AUC (macro) | Macro-F1 | Accuracy | Baseline Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline (Luôn đoán Lớp 0)** | 0,5000 | 0,3333 | 0,3048 | 0,8424 | 0,8424 |
| **Logistic Regression (Hồi quy Logistic)** | 0,7763 | 0,4643 | 0,4620 | 0,7984 | 0,8424 |
| **XGBoost (Cây quyết định tăng cường)** | 0,7858 | 0,4741 | 0,4654 | 0,8057 | 0,8424 |
| **MLP (Mạng nơ-ron nhiều tầng — ĐƯỢC CHỌN)** | 0,7879 | 0,4743 | 0,4738 | 0,8069 | 0,8424 |
<!-- /AUTO -->

### Hiệu năng trên tập Kiểm thử độc lập (Test — Đánh giá một lần duy nhất)

Kết quả đo đạc chính thức trên tập Test sau khi đã cố định toàn bộ siêu tham số và mô hình:

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

### Hiệu năng trên bài toán phân loại nhị phân (Binary)

Đối chứng khả năng phân loại nhị phân (gộp nhóm tiền tiểu đường vào nhóm có nguy cơ hoặc tách biệt lớp nhãn):

<!-- AUTO:binary_results -->
| Mô hình | ROC-AUC (95% CI nếu có) | PR-AUC macro | Macro-F1 | Accuracy | Recall Lớp 1 (Có bệnh) | Baseline Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Majority Baseline | 0,5000 | 0,5000 | 0,4572 | 0,8424 | 0,0000 | 0,8424 |
| Logistic Regression | 0,8153 | 0,6861 | 0,6772 | 0,8166 | 0,5058 | 0,8424 |
| XGBoost | 0,8244 | 0,7101 | 0,6867 | 0,8163 | 0,5493 | 0,8424 |
| MLP | 0,8231 | 0,7012 | 0,6854 | 0,8176 | 0,5376 | 0,8424 |
<!-- /AUTO -->

---

## Hiệu chuẩn

Mô hình học máy nguyên bản (raw logits/probabilities) thường gặp tình trạng mất tự tin hoặc quá tự tin vào xác suất đầu ra. Trong dự án này:
- **Phương pháp lựa chọn:** Hiệu chuẩn xác suất được lựa chọn thông qua phương pháp Cross-Validation dạng Out-of-fold (OOF) độc quyền trên tập Validation nhằm hạn chế tối đa tính lạc quan ảo, sau đó áp dụng cố định lên tập Test.
- **Tiêu chí đánh giá:** Sai số hiệu chuẩn kỳ vọng (ECE — Expected Calibration Error) và điểm Brier đa lớp.

<!-- AUTO:calibration -->
| Tiêu chí đo lường hiệu chuẩn | Trước hiệu chuẩn (Thô / Raw) | Sau hiệu chuẩn (Calibrated) | Cải thiện / Ý nghĩa |
| :--- | :--- | :--- | :--- |
| Phương pháp hiệu chuẩn | — | **isotonic_ovr** | Lựa chọn qua Out-of-fold cross-validation trên tập Val |
| ECE Macro trên tập Test | **0,0033** | **0,0024** | Giảm sai số hiệu chuẩn xác suất toàn thể |
| Brier Score đa lớp trên Test | **0,2212** | **0,2213** | Duy trì độ sắc bén dự báo |
| ECE Lớp 0 (Không tiểu đường) | — | **0,0036** | Xác suất phân bổ sát tần suất thực nghiệm |
| ECE Lớp 1 (Tiền tiểu đường) | — | **0,0002** | Rất thấp do dự đoán thận trọng ở lớp thiểu số |
| ECE Lớp 2 (Tiểu đường) | — | **0,0033** | Xác suất tin cậy trực tiếp cho phân tầng nguy cơ |
<!-- /AUTO -->

---

## Dòng trùng lặp và đối chứng

Do bản chất dữ liệu BRFSS gồm các câu hỏi trắc nghiệm rời rạc, xuất hiện hiện tượng nhiều người tham gia có cùng một hồ sơ câu trả lời (identical survey response profiles). Để kiểm soát nguy cơ rò rỉ thông tin hoặc việc mô hình ghi nhớ máy móc các dòng trùng lặp, dự án thực hiện hai đối chứng nghiêm ngặt:

1. **Đối chứng trên hai biến thể dữ liệu:** So sánh mô hình huấn luyện trên bản đầy đủ (Full) và mô hình huấn luyện trên bản đã loại bỏ trùng lặp (Dedup).
2. **Đánh giá trên tập con hoàn toàn mới (Novel / No-twins subset):** Chỉ đánh giá trên các dòng trong tập Test mà tổ hợp đặc trưng của chúng chưa từng xuất hiện trong tập Train.

<!-- AUTO:control_dedup -->
| Chỉ số hiệu năng (Mô hình MLP) | Bản đầy đủ (Full, N=50.736) | Bản loại trùng (Dedup, N=45.957) | Chênh lệch (Full - Dedup) |
| :--- | :--- | :--- | :--- |
| Tỉ lệ mẫu trùng lặp chéo với Train | **12,75%** (6.469 mẫu) | **1,14%** (522 mẫu) | -11,61 điểm % |
| ROC-AUC macro | **0,7858** [0,7797; 0,7923] | **0,7742** [0,7676; 0,7808] | +0,0117 |
| Macro-F1 | **0,4641** | **0,4655** | -0,0014 |
| Accuracy | **0,8050** | **0,7830** | +0,0220 |
| Baseline Accuracy | **0,8424** | **0,8271** | — |
<!-- /AUTO -->

<!-- AUTO:novel_subset -->
| Mô hình đánh giá | ROC-AUC trên toàn tập Test (N=50.736) | ROC-AUC trên tập Novel / No-twins (N=44.267, 87,25%) | Độ suy giảm AUC |
| :--- | :--- | :--- | :--- |
| Majority Baseline | 0,5000 | 0,5000 | 0,0000 |
| Logistic Regression | 0,7719 | 0,7603 | 0,0116 |
| XGBoost | 0,7873 | 0,7763 | 0,0110 |
| **MLP (Mô hình chính)** | **0,7858** | **0,7750** | **0,0108** |
<!-- /AUTO -->

---

## Mức nguy cơ và cảnh báo

Hệ thống chuyển đổi xác suất dự đoán của lớp bệnh tiểu đường thành các tầng nguy cơ phục vụ định hướng sàng lọc cộng đồng:
- **Nguy cơ Trung bình:** Ngưỡng xác suất được tối ưu để tối đa hóa độ nhạy (Recall), nhằm hạn chế tối đa việc bỏ sót người có nguy cơ trong sàng lọc ban đầu.
- **Nguy cơ Cao:** Ngưỡng xác suất chặt chẽ hơn nhằm nâng cao giá trị dự đoán dương tính (PPV) và độ nâng (Lift).

<!-- AUTO:risk_alert -->
| Tầng cảnh báo nguy cơ | Ngưỡng xác suất ($p$) | Tỉ lệ phát cảnh báo (Alert Rate) | Số ca cảnh báo | Độ nhạy / Bắt ca bệnh (Recall) | PPV (Độ chính xác cảnh báo) | Tỉ lệ dương tính giả (FPR) | Độ nâng so với nền (Lift) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tỉ lệ nền (Toàn tập test)** | — | 100,00% | 50.736 | 100,00% | 13,93% | — | 1,00x |
| **Nguy cơ Trung bình trở lên** | $p \ge 0,10$ | **49,88%** | 25.309 | **90,44%** (6.393/7.069) | **25,26%** | 43,32% | **1,81x** |
| **Nguy cơ Cao** | $p \ge 0,25$ | **22,19%** | 11.257 | **60,55%** (4.280/7.069) | **38,02%** | 15,98% | **2,73x** |
<!-- /AUTO -->

---

## Nhóm bị bỏ sót

Dựa trên số liệu đo đạc thực tế, nhóm đối tượng bị hệ thống bỏ sót nhiều nhất là:
1. **Người mắc tiền tiểu đường (Lớp 1):** Độ nhạy (Recall) đối với lớp này cực kỳ thấp như thể hiện trong bảng số liệu. Ranh giới giữa người khỏe mạnh và người tiền tiểu đường qua các câu hỏi hành vi lối sống thông thường là không rõ ràng.
2. **Người trẻ tuổi (dưới 35 tuổi) có nguy cơ tiềm ẩn:** Do tỉ lệ hiện mắc tiểu đường ở người trẻ trong dữ liệu khảo sát rất thấp, mô hình có xu hướng dự đoán xác suất thấp cho nhóm này, dẫn đến độ nhạy phát hiện bệnh ở người trẻ thấp hơn nhóm lớn tuổi.
3. **Các cá nhân có bệnh lý thứ phát hoặc thể bệnh không liên quan đến béo phì:** Những người có chỉ số BMI bình thường nhưng có nguy cơ do di truyền hoặc cơ chế tự miễn (tiểu đường tuýp 1, LADA) có thể bị mô hình bỏ sót do mô hình phụ thuộc nhiều vào các yếu tố nguy cơ truyền thống như BMI và huyết áp.

---

## Cách dùng đúng

- Dùng trong môi trường nghiên cứu, hội thảo học thuật hoặc làm ví dụ giảng dạy về quy trình xây dựng AI y tế đáng tin cậy.
- Dùng để hiểu cách thức các yếu tố nguy cơ dịch tễ học (huyết áp, thể trạng, vận động) tương quan thống kê với tình trạng sức khỏe cộng đồng.
- Dùng làm công cụ minh họa trực quan trong giáo dục nâng cao nhận thức sức khỏe: giúp người dùng nhận biết các thói quen sinh hoạt tốt và xấu có thể ảnh hưởng như thế nào đến ước lượng nguy cơ.

---

## Cách dùng sai (không dùng để chẩn đoán)

- **TUYỆT ĐỐI KHÔNG DÙNG ĐỂ CHẨN ĐOÁN BỆNH:** Không được dựa vào kết quả dự đoán của mô hình để tự kết luận mình có mắc bệnh tiểu đường hay không.
- **Không dùng để tự kê đơn, thay đổi liều lượng thuốc hoặc bỏ qua chỉ định của bác sĩ.**
- **Không dùng thay thế cho các xét nghiệm cận lâm sàng chuẩn:** Chẩn đoán tiểu đường bắt buộc phải thông qua xét nghiệm máu chuyên khoa (đường huyết lúc đói, nghiệm pháp dung nạp glucose, hoặc chỉ số HbA1c).
- **Không dùng trong môi trường cấp cứu hoặc làm căn cứ phê duyệt hồ sơ bảo hiểm y tế.**

---

## Hạn chế

1. **Nhóm tiền tiểu đường gần như không thể phân biệt:** Do thiếu các chỉ số sinh hóa đặc hiệu (xét nghiệm máu) và sự mất cân bằng dữ liệu trầm trọng, mô hình không thể phát hiện hiệu quả nhóm tiền tiểu đường chỉ qua dữ liệu hành vi.
2. **Accuracy thấp hơn baseline "luôn đoán không" và lý do không dùng Accuracy:**
   - Trong tập dữ liệu này, nhóm không mắc bệnh chiếm đa số áp đảo. Nếu một mô hình tầm thường chỉ đoán "không mắc bệnh" cho tất cả mọi người, nó sẽ đạt độ chính xác (Accuracy) rất cao theo tỷ lệ lớp đa số, nhưng hoàn toàn vô dụng vì bỏ sót toàn bộ người bệnh (Recall bằng 0).
   - Mô hình của dự án ưu tiên phát hiện người bệnh (tối ưu Recall và ROC-AUC), chấp nhận hy sinh Accuracy tổng thể để đạt độ nhạy cao. Do đó, Accuracy của mô hình thấp hơn baseline và hoàn toàn không phải là thước đo phù hợp cho bài toán y tế mất cân bằng nhãn.
3. **Hiệu chuẩn và ngưỡng cùng chọn trên tập Validation:** Mặc dù đã dùng kỹ thuật Out-of-fold để giảm thiểu hiện tượng quá lạc quan, việc chọn đồng thời cả phương pháp hiệu chuẩn và ngưỡng quyết định trên cùng một tập dữ liệu validation vẫn có thể mang lại một mức độ sai số nhỏ khi suy rộng.
4. **Bản chất tương quan thống kê trên khảo sát Hoa Kỳ 2015:** Dữ liệu bắt nguồn từ một cuộc khảo sát dân số tại Mỹ vào năm 2015 qua điện thoại. Mọi kết quả chỉ phản ánh mối liên hệ **tương quan thống kê** trong mẫu điều tra đó, hoàn toàn **không phản ánh quan hệ nhân quả y khoa** và không đại diện cho cấu trúc nhân khẩu học hay bệnh lý của các quốc gia khác (như Việt Nam).

---

> **MIỄN TRỪ TRÁCH NHIỆM CUỐI TÀI LIỆU:**
> Mô hình và toàn bộ tài liệu này được công bố **chỉ để học tập, không phải chẩn đoán**. Mọi quyết định liên quan đến sức khỏe cá nhân cần được tham vấn trực tiếp với nhân viên y tế có chuyên môn.
