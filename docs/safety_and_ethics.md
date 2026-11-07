# Khung An Toàn, Đạo Đức và Trách Nhiệm Y Tế

> **Miễn trừ trách nhiệm:** Hệ thống này được xây dựng **chỉ để học tập, nghiên cứu và thử nghiệm công nghệ; không phải chẩn đoán y khoa**. Mô hình không thể và không được dùng để thay thế ý kiến chuyên môn của bác sĩ, nhân viên y tế hoặc các xét nghiệm y tế chính thức.

---

## 1. Nguyên tắc không chẩn đoán và khuyến cáo xét nghiệm xác nhận

- **Không chẩn đoán:** Kết quả phân lớp và các điểm xác suất do mô hình cung cấp phản ánh mức độ nguy cơ thống kê dựa trên tập dữ liệu khảo sát cộng đồng. Đây hoàn toàn **không phải chẩn đoán lâm sàng**.
- **Luôn khuyên xét nghiệm xác nhận:** Mọi cá nhân nhận cảnh báo nguy cơ từ hệ thống, hoặc có bất kỳ triệu chứng nghi ngờ nào liên quan đến rối loạn đường huyết, cần đến các cơ sở y tế chuyên khoa để thực hiện **xét nghiệm xác nhận** tiêu chuẩn vàng, bao gồm:
  - Xét nghiệm **đường huyết đói hoặc HbA1c** (Fasting Plasma Glucose $\ge 126\text{ mg/dL}$ hoặc $\text{HbA1c} \ge 6.5\%$).
  - Nghiệm pháp dung nạp glucose đường uống (OGTT 2 giờ $\ge 200\text{ mg/dL}$) khi có chỉ định của bác sĩ.
- Không tự ý bắt đầu, thay đổi liều lượng hoặc ngưng sử dụng bất kỳ loại thuốc điều trị nào dựa trên kết quả đầu ra của hệ thống này.

---

## 2. Hệ thống luật an toàn lâm sàng (Clinical Safety Rules)

Để bảo đảm tính an toàn tối đa cho người sử dụng, hệ thống tích hợp bộ luật an toàn độc lập với các nguyên tắc hoạt động nghiêm ngặt:

1. **Ưu tiên cao hơn mô hình máy học:**
   - Khi phát hiện các dấu hiệu nguy cơ cấp bách (cờ đỏ triệu chứng hoặc chỉ số sinh trắc học cực đoan), luật an toàn sẽ tự động kích hoạt và **ghi đè (override) dự đoán của mô hình máy học**, nâng mức đánh giá lên nguy cơ cao hoặc khuyên khám y tế khẩn cấp.
   - Ví dụ: Người dùng khai báo các triệu chứng cờ đỏ kinh điển (sút cân không rõ nguyên nhân, khát nước liên tục, đa niệu) hoặc có chỉ số $\text{BMI} \ge 45$ kèm đồng thời tiền sử tăng huyết áp và tăng mỡ máu.
2. **Chỉ chạy khi có đầy đủ dữ liệu:**
   - Luật an toàn tuân thủ nguyên tắc thận trọng: **chỉ kích hoạt khi trường dữ liệu được cung cấp rõ ràng**.
   - Nếu dữ liệu đầu vào bị khuyết thiếu, hệ thống tuyệt đối không tự động gán giá trị mặc định để suy diễn sai lệch hay đưa ra cảnh báo không có căn cứ thực tế.

---

## 3. Thiên kiến dữ liệu (Dataset Bias)

Người sử dụng và các bên liên quan cần nhận thức rõ các giới hạn vốn có của dữ liệu huấn luyện:

- **Nguồn dữ liệu:** Mô hình được huấn luyện trên bộ dữ liệu BRFSS 2015 (Behavioral Risk Factor Surveillance System) do Trung tâm Kiểm soát và Phòng ngừa Dịch bệnh Hoa Kỳ (CDC) thu thập qua khảo sát phỏng vấn điện thoại.
- **Phương thức tự khai (Self-reported):** Hầu hết các biến số (chiều cao, cân nặng để tính BMI, tiền sử huyết áp, cholesterol, mức độ vận động) đều do đối tượng tham gia tự khai báo. Phương thức này tồn tại sai số nhớ lại (recall bias), sự thiếu chính xác trong ước lượng bản thân và không thể so sánh với các phép đo sinh học trực tiếp.
- **Tính đại diện nhân khẩu học:** Dữ liệu phản ánh đặc điểm dịch tễ học của người trưởng thành tại Hoa Kỳ vào năm 2015. Mô hình **không đại diện cho dân số toàn cầu hoặc quần thể người Việt Nam**, nơi có ngưỡng BMI cảnh báo thừa cân/béo phì thấp hơn (theo chuẩn WHO Tây Thái Bình Dương) và chế độ dinh dưỡng, lối sống rất khác biệt.

---

## 4. Chênh lệch giữa các nhóm đối tượng (Fairness & Disparities)

Qua quá trình phân tích công bằng đa chiều (theo Độ tuổi, Giới tính, Mức thu nhập):

- **Chênh lệch tỷ lệ phát hiện (Recall) và tỷ lệ dương tính giả (FPR):** Mô hình có xu hướng đạt độ nhạy (Recall) cao hơn ở nhóm người cao tuổi ($\ge 65$ tuổi) nhưng đi kèm tỷ lệ dương tính giả (FPR) tăng cao, do tỷ lệ nền mắc bệnh ở nhóm tuổi này trong xã hội lớn hơn đáng kể.
- **Chênh lệch theo điều kiện kinh tế:** Các nhóm thu nhập thấp có tỷ lệ cảnh báo nguy cơ cao hơn, phản ánh sự bất bình đẳng trong tiếp cận thực phẩm lành mạnh và điều kiện chăm sóc sức khỏe ban đầu được ghi nhận trong khảo sát BRFSS.
- **Cam kết trung thực:** Dự án ghi nhận trung thực các khoảng chênh lệch hiệu năng thông qua khoảng tin cậy Wilson 95%. **Dự án không áp dụng các biện pháp can thiệp giả tạo để giảm chênh lệch** vì việc gượng ép cân bằng toán học nhân tạo có thể bóp méo ranh giới phân loại y khoa thực tế.

---

## 5. Quyền riêng tư và bảo mật dữ liệu người dùng

- **Thiết kế phi trạng thái (Stateless by Design):** Toàn bộ hệ thống API phục vụ (`FastAPI`) và giao diện web người dùng hoạt động theo cơ chế phi trạng thái.
- **Không lưu trữ dữ liệu cá nhân:** Hệ thống **không thu thập, không ghi nhật ký và không lưu trữ** bất kỳ thông tin định danh cá nhân nào (PII - họ tên, số điện thoại, địa chỉ, định danh thiết bị).
- **Không lưu chỉ số sức khỏe:** Dữ liệu chỉ số sức khỏe gửi lên chỉ được xử lý tạm thời trên bộ nhớ đệm (RAM) trong quá trình tính toán phản hồi và được giải phóng ngay sau khi phản hồi trả về client. Không có bất kỳ cơ sở dữ liệu (database) nào được sử dụng để tích lũy dữ liệu người dùng.

---

## 6. Đối tượng không nên sử dụng

Hệ thống không được thiết kế cho các nhóm đối tượng sau:

1. **Trẻ em và thanh thiếu niên dưới 18 tuổi:** Dữ liệu huấn luyện chỉ thu thập trên người trưởng thành.
2. **Phụ nữ đang mang thai:** Đái tháo đường thai kỳ (Gestational Diabetes) có cơ chế bệnh sinh, sinh lý bệnh và tiêu chuẩn chẩn đoán hoàn toàn tách biệt so với đái tháo đường typ 2 ở người lớn.
3. **Người mắc đái tháo đường typ 1:** Đái tháo đường typ 1 là bệnh lý tự miễn phá hủy tế bào beta đảo tụy, không thể sàng lọc qua các yếu tố nguy cơ lối sống của khảo sát BRFSS.
4. **Tình trạng cấp cứu y tế:** Người đang có dấu hiệu cấp cứu chuyển hóa (hôn mê, thở dốc, lú lẫn, sốt cao mất nước) cần liên hệ cấp cứu y tế 115 hoặc đến ngay bệnh viện gần nhất, không được sử dụng công cụ trực tuyến.
5. **Người tìm kiếm kết luận chẩn đoán xác định:** Như đã khẳng định, hệ thống **chỉ để học tập, không phải chẩn đoán** y khoa.
