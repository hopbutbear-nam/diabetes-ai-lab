# Kho tri thức Y khoa Đái tháo đường (Knowledge Base)

Tài liệu tri thức chuẩn hóa phục vụ tra cứu ngữ cảnh lâm sàng bằng Graph-RAG cho hệ thống hỗ trợ phòng ngừa và sàng lọc đái tháo đường.

## 1. Nguồn tài liệu gốc và ngày truy cập

Tài liệu được tổng hợp từ hai cơ quan y tế uy tín hàng đầu:

1. **Hiệp hội Đái tháo đường Hoa Kỳ (American Diabetes Association - ADA)**
   - Văn bản: *Standards of Care in Diabetes — 2024* (Tiêu chuẩn Chăm sóc Y tế trong Đái tháo đường).
   - Xuất bản: *Diabetes Care*, Tập 47, Phụ bản 1, tháng 1 năm 2024.
   - Các chương tham chiếu chính:
     - Section 2: *Classification and Diagnosis of Diabetes* (Phân loại và chẩn đoán).
     - Section 5: *Facilitating Positive Health Behaviors and Well-being* (Hành vi sức khỏe và lối sống).
     - Section 8: *Obesity and Weight Management for the Prevention and Treatment of Type 2 Diabetes* (Quản lý béo phì).
     - Section 10: *Cardiovascular Disease and Risk Management* (Bệnh tim mạch và quản lý nguy cơ).
     - Section 11 & 12: *Microvascular Complications and Foot Care* (Biến chứng vi mạch và bàn chân).
   - Ngày truy cập: 05/11/2026.

2. **Viện Quốc gia về Bệnh tiểu đường, Tiêu hóa và Bệnh thận Hoa Kỳ (NIDDK - NIH)**
   - Chuyên mục: *Diabetes Symptoms, Causes & Risk Factors*.
   - Địa chỉ trực tuyến: https://www.niddk.nih.gov/health-information/diabetes
   - Ngày truy cập: 05/11/2026.

## 2. Giấy phép và mục đích sử dụng

- **Mục đích:** Nghiên cứu khoa học, học thuật và thực nghiệm công nghệ hỗ trợ AI phi thương mại.
- **Giấy phép:** Tài liệu tham khảo y khoa công khai. Toàn bộ nội dung trích lục được giữ nguyên nguồn tham chiếu và không dùng vào mục đích phân phối thương mại độc quyền.

## 3. Quy cách biên soạn bộ dữ liệu mẫu

> [!NOTE]
> **Bộ mẫu biên soạn tay (Hand-curated sample corpus):**
> Bộ dữ liệu mẫu gồm 7 đoạn văn bản được biên soạn tay trực tiếp từ các tiêu chuẩn chẩn đoán chính thức của ADA 2024 và NIDDK, đảm bảo tính súc tích, chính xác tuyệt đối về ngưỡng số liệu y khoa.

- **Độ dài đoạn văn:** Mỗi đoạn được cắt chuẩn hóa trong khoảng từ 120 đến 200 từ tiếng Việt.
- **Các ngưỡng y khoa then chốt trong bộ mẫu:**
  - **Đái tháo đường:** Đường huyết đói (FPG) $\ge$ 126 mg/dL (7,0 mmol/L); HbA1c $\ge$ 6,5%; Nghiệm pháp OGTT 2 giờ $\ge$ 200 mg/dL; Đường huyết bất kỳ $\ge$ 200 mg/dL kèm triệu chứng.
  - **Tiền đái tháo đường:** Đường huyết đói 100–125 mg/dL; OGTT 2 giờ 140–199 mg/dL; HbA1c 5,7%–6,4%.
  - **Huyết áp mục tiêu:** Dưới 130/80 mmHg.
  - **Chỉ số khối cơ thể (BMI):** Sàng lọc từ BMI $\ge$ 23 kg/m² đối với người châu Á hoặc $\ge$ 25 kg/m² nói chung; giảm 5–7% cân nặng.
  - **Vận động:** Tối thiểu 150 phút/tuần bài tập aerobic cường độ trung bình kèm bài tập kháng lực.

## 4. Cấu trúc đồ thị tri thức (Graph Schema)

Đồ thị bao gồm 11 nút khái niệm y khoa (`dai_thao_duong`, `tien_dai_thao_duong`, `duong_huyet_doi`, `hba1c`, `dung_nap_glucose`, `trieu_chung`, `huyet_ap`, `tim_mach`, `bmi`, `bien_chung`, `loi_song`) và 15 quan hệ có hướng (tiêu chuẩn chẩn đoán, yếu tố nguy cơ, tiến triển, biến chứng, kiểm soát).

## 5. Tuyên bố miễn trừ trách nhiệm y tế (Disclaimer)

Toàn bộ thông tin trong kho tri thức này chỉ nhằm mục đích nghiên cứu thử nghiệm và hỗ trợ giáo dục thông tin. Dữ liệu này **không cấu thành lời khuyên y khoa, không thay thế chẩn đoán chuyên khoa hoặc chỉ định điều trị của bác sĩ**. Mọi trường hợp có triệu chứng bất thường cần đến cơ sở y tế để được thăm khám trực tiếp.
