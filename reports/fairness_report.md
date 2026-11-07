# Báo cáo đánh giá công bằng (Fairness Report) — Hệ thống sàng lọc nguy cơ tiểu đường

> **MIỄN TRỪ TRÁCH NHIỆM:**
> Báo cáo này được thực hiện trong khuôn khổ dự án nghiên cứu kỹ thuật và học tập. Tài liệu này **chỉ để học tập, không phải chẩn đoán y tế**.

---

## 1. Mục đích và phạm vi đánh giá

Trong các hệ thống học máy áp dụng trong y tế và chăm sóc sức khỏe cộng đồng, việc đánh giá tính công bằng (Fairness & Bias) giữa các nhóm nhân khẩu học khác nhau là bắt buộc nhằm:
- Nhận diện các nhóm dân số có nguy cơ bị bỏ sót ca bệnh cao (độ nhạy/recall thấp).
- Đánh giá sự khác biệt về tỉ lệ cảnh báo (alert rate) và độ chính xác dự báo dương tính (precision/PPV).
- Đảm bảo tính minh bạch khoa học, không che giấu sự chênh lệch hiệu năng giữa các tầng lớp xã hội.

Các thuộc tính nhân khẩu học được đánh giá bao gồm:
1. **Độ tuổi (Age):** 6 nhóm tuổi từ thanh niên đến người cao tuổi (18–34, 35–44, 45–54, 55–64, 65–74, 75+).
2. **Mức thu nhập hộ gia đình (Income):** 3 nhóm thu nhập (Dưới $25.000, $25.000 – $50.000, Trên $50.000).
3. **Giới tính sinh học (Sex):** Nam giới và Nữ giới.

---

## 2. Phương pháp luận đánh giá

- **Tập dữ liệu kiểm thử:** Đánh giá độc lập trên tập Test (N = 50.736 mẫu) của biến thể dữ liệu đầy đủ (Full).
- **Mô hình đánh giá:** Mô hình Mạng nơ-ron nhiều tầng (MLP) sau khi đã qua hiệu chuẩn xác suất và áp dụng ngưỡng phân tầng nguy cơ.
- **Tiêu chí số lượng mẫu dương tối thiểu (`min_pos = 30`):**
  - Nhằm đảm bảo ý nghĩa thống kê của các ước lượng tỉ lệ, quy định mỗi phân nhóm bắt buộc phải có ít nhất 30 ca dương tính thực tế ($N_{pos} \ge 30$).
  - Nếu một phân nhóm có ít hơn 30 ca dương, hệ thống sẽ tự động bật cờ cảnh báo mẫu nhỏ (`small_sample = True`) để nhắc nhở người đọc không kết luận vội vàng.
- **Khoảng tin cậy Wilson (Wilson Score Interval 95%):**
  - Mọi chỉ số tỉ lệ (Prevalence, Recall, Precision, Alert Rate) đều được tính kèm khoảng tin cậy 95% theo công thức Wilson Score Interval.
  - Phương pháp Wilson đặc biệt phù hợp và chính xác hơn khoảng tin cậy chuẩn tắc (Wald) đối với các phân nhóm có tỉ lệ sự kiện thấp hoặc cỡ mẫu hạn chế.
- **Thước đo độ chênh lệch cực đại (`max_gap_pp`):**
  - Đo bằng điểm phần trăm (percentage points - pp) giữa nhóm có Recall cao nhất và nhóm có Recall thấp nhất trong cùng một thuộc tính nhân khẩu học:
  $$\Delta_{\text{Recall}} = \max(\text{Recall}) - \min(\text{Recall})$$

---

## 3. Kết quả đánh giá theo phân nhóm

Toàn bộ các bảng số liệu dưới đây được trích xuất tự động từ hệ thống đo lường:

<!-- AUTO:fairness_age -->
### Đánh giá công bằng theo Độ tuổi (Age) (Chênh lệch tối đa: **42,13 pp**)

| Phân nhóm | Số mẫu (N) | Ca bệnh (N pos) | Tỉ lệ nền (Prevalence) | Recall (95% Wilson CI) | Precision (95% Wilson CI) | Tỉ lệ cảnh báo (Alert Rate) | Cờ mẫu nhỏ (<30 ca) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 65–74 tuổi | 11.073 | 2.370 | 21,40% | **57,97%** [55,98%; 59,95%] | 44,25% [42,51%; 46,00%] | 34,43% | Đạt chuẩn |
| 45–54 tuổi | 9.231 | 963 | 10,43% | **42,26%** [39,18%; 45,41%] | 40,54% [37,54%; 43,61%] | 14,43% | Đạt chuẩn |
| 35–44 tuổi | 6.086 | 339 | 5,57% | **24,78%** [20,48%; 29,64%] | 37,67% [31,57%; 44,19%] | 5,34% | Đạt chuẩn |
| 75 tuổi trở lên | 6.608 | 1.303 | 19,72% | **51,50%** [48,78%; 54,20%] | 36,57% [34,39%; 38,80%] | 35,41% | Đạt chuẩn |
| 55–64 tuổi | 12.898 | 1.993 | 15,45% | **56,85%** [54,66%; 59,01%] | 41,35% [39,52%; 43,21%] | 26,23% | Đạt chuẩn |
| 18–34 tuổi | 4.840 | 101 | 2,09% | **15,84%** [9,99%; 24,19%] | 51,61% [34,84%; 68,03%] | 1,34% | Đạt chuẩn |
<!-- /AUTO -->

<!-- AUTO:fairness_income -->
### Đánh giá công bằng theo Thu nhập (Income) (Chênh lệch tối đa: **27,84 pp**)

| Phân nhóm | Số mẫu (N) | Ca bệnh (N pos) | Tỉ lệ nền (Prevalence) | Recall (95% Wilson CI) | Precision (95% Wilson CI) | Tỉ lệ cảnh báo (Alert Rate) | Cờ mẫu nhỏ (<30 ca) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Trên $50.000 | 26.762 | 2.535 | 9,47% | **38,03%** [36,16%; 39,93%] | 39,61% [37,68%; 41,56%] | 11,96% | Đạt chuẩn |
| Dưới $25.000 | 11.491 | 2.619 | 22,79% | **65,86%** [64,03%; 67,66%] | 44,16% [42,61%; 45,73%] | 41,22% | Đạt chuẩn |
| $25.000 – $50.000 | 12.483 | 1.915 | 15,34% | **52,01%** [49,77%; 54,24%] | 38,34% [36,49%; 40,22%] | 26,58% | Đạt chuẩn |
<!-- /AUTO -->

<!-- AUTO:fairness_sex -->
### Đánh giá công bằng theo Giới tính (Sex) (Chênh lệch tối đa: **1,82 pp**)

| Phân nhóm | Số mẫu (N) | Ca bệnh (N pos) | Tỉ lệ nền (Prevalence) | Recall (95% Wilson CI) | Precision (95% Wilson CI) | Tỉ lệ cảnh báo (Alert Rate) | Cờ mẫu nhỏ (<30 ca) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Nam giới | 22.527 | 3.401 | 15,10% | **53,07%** [51,39%; 54,75%] | 40,28% [38,85%; 41,72%] | 24,86% | Đạt chuẩn |
| Nữ giới | 28.209 | 3.668 | 13,00% | **51,25%** [49,64%; 52,87%] | 42,18% [40,74%; 43,64%] | 20,05% | Đạt chuẩn |
<!-- /AUTO -->

---

## 4. Đọc kết quả

Khi phân tích các bảng số liệu đo đạc tự động ở trên, cần lưu ý các khía cạnh dịch tễ và thống kê sau:

1. **Chênh lệch Recall gắn liền với sự khác biệt về tỉ lệ nền (Prevalence):**
   - Giữa các nhóm tuổi, độ nhạy (Recall) tăng dần theo độ tuổi: nhóm thanh niên (18–34 tuổi) có Recall thấp nhất, trong khi nhóm người cao tuổi (65–74 tuổi) đạt Recall cao nhất.
   - Nguyên nhân kỹ thuật chính là **tỉ lệ mắc bệnh nền (Prevalence) rất khác nhau giữa các nhóm**. Trong dữ liệu khảo sát, tỉ lệ người trẻ mắc tiểu đường rất thấp (dưới 3%), trong khi ở người cao tuổi tỉ lệ này lên đến hơn 20%. Do mô hình tối ưu trên toàn quần thể, các đặc trưng thể hiện ở nhóm người trẻ không đủ mạnh để vượt qua ngưỡng xác suất phân loại, dẫn đến tỉ lệ cảnh báo và Recall ở nhóm trẻ thấp hơn đáng kể.
   - Tương tự, đối với nhóm thu nhập, nhóm có thu nhập dưới $25.000 có tỉ lệ mắc bệnh nền cao hơn và cũng đạt Recall cao hơn so với nhóm có thu nhập trên $50.000.
2. **Sự đồng đều giữa hai giới tính:**
   - Chênh lệch Recall giữa Nam và Nữ là rất nhỏ (khoảng cách điểm phần trăm dưới 2 pp), cho thấy mô hình không bị thiên lệch đáng kể đối với yếu tố giới tính sinh học.
3. **Không có giải pháp giảm chênh lệch trong dự án này:**
   - **Dự án này chủ động không áp dụng bất kỳ kỹ thuật can thiệp giảm chênh lệch nhân tạo nào** (như cân bằng ngưỡng riêng cho từng nhóm - demographic parity thresholds, hay điều chỉnh xác suất hậu kỳ theo nhóm nhân khẩu).
   - **Lý do:** Trong bài toán ước lượng nguy cơ y tế cá nhân hóa, việc can thiệp nhân tạo nhằm ép buộc Recall của nhóm trẻ tuổi bằng với nhóm cao tuổi sẽ làm méo mó nghiêm trọng xác suất đã được hiệu chuẩn (calibrated risk probability), dẫn đến việc phát sinh một lượng khổng lồ các ca dương tính giả ở người trẻ, làm mất uy tín của hệ thống cảnh báo và gây lãng phí nguồn lực y tế. Thay vào đó, dự án lựa chọn phương pháp **công khai minh bạch hoàn toàn các hạn chế** để bác sĩ và người dùng hiểu rõ phạm vi tin cậy của mô hình.

---

## 5. Hạn chế

1. **Nguồn dữ liệu tự khai và thiên lệch khảo sát:**
   - Dữ liệu BRFSS 2015 được thu thập qua các cuộc gọi điện thoại tại Hoa Kỳ. Nhóm người có thu nhập rất thấp, người vô gia cư hoặc người không có điện thoại có thể không được đại diện đầy đủ.
   - Các câu trả lời về chẩn đoán bệnh tật, chiều cao, cân nặng (để tính BMI) đều là tự báo cáo, tiềm ẩn sai lệch hồi tưởng (recall bias) hoặc tâm lý khai báo thiên lệch xã hội (social desirability bias).
2. **Không phản ánh mối quan hệ nhân quả:**
   - Mối tương quan giữa thu nhập, tuổi tác và nguy cơ tiểu đường phản ánh tình trạng bất bình đẳng sức khỏe xã hội tại Mỹ năm 2015, hoàn toàn không phải là quy luật sinh học nhân quả trực tiếp.
3. **Không thể suy rộng sang quần thể khác:**
   - Cấu trúc nhân khẩu học và phân bố bệnh tật tại Việt Nam hoàn toàn khác biệt so với Hoa Kỳ năm 2015. Tuyệt đối không áp dụng kết quả của báo cáo này vào thực tế lâm sàng tại Việt Nam.

---

> **MIỄN TRỪ TRÁCH NHIỆM CUỐI TÀI LIỆU:**
> Tài liệu này được tạo ra **chỉ để học tập, không phải chẩn đoán**. Mọi quyết định chăm sóc sức khỏe cần được tư vấn bởi các chuyên gia y tế chuyên môn.
