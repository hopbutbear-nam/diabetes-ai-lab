# Dữ liệu

| Bộ | Nguồn | Vai trò |
|---|---|---|
| BRFSS 2015 (Diabetes Health Indicators) | CDC / Kaggle: "Diabetes Health Indicators Dataset" | Chính: 253.680 hồ sơ, 21 đặc trưng, nhãn 3 lớp |
| NHANES 2011-2018 | https://wwwn.cdc.gov/nchs/nhanes/ | Có xét nghiệm (glucose, HbA1c), đối chiếu |
| Pima Indians Diabetes | UCI / Kaggle | Nhỏ (768), dùng để học và kiểm tra pipeline |

Đặt file tải về vào `data/raw/<ten-bo>/`. Thư mục này bị `.gitignore`, **không commit dữ liệu**.
Kiểm tra giấy phép từng bộ trước khi dùng. Ghi lại ngày tải và phiên bản trong `docs/data_dictionary.md`.
