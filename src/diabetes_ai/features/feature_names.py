"""Tên tiếng Việt và nhãn danh mục cho từng đặc trưng BRFSS."""

VI_FEATURE: dict[str, str] = {
    "HighBP": "Huyết áp cao",
    "HighChol": "Mỡ máu cao",
    "CholCheck": "Kiểm tra cholesterol trong 5 năm",
    "BMI": "Chỉ số BMI",
    "Smoker": "Hút thuốc (≥100 điếu)",
    "Stroke": "Từng bị đột quỵ",
    "HeartDiseaseorAttack": "Bệnh tim mạch hoặc nhồi máu cơ tim",
    "PhysActivity": "Hoạt động thể chất (30 ngày qua)",
    "Fruits": "Ăn trái cây hàng ngày",
    "Veggies": "Ăn rau hàng ngày",
    "HvyAlcoholConsump": "Uống nhiều rượu bia",
    "AnyHealthcare": "Có bảo hiểm y tế",
    "NoDocbcCost": "Không thể khám bệnh vì chi phí",
    "GenHlth": "Tự đánh giá sức khỏe (1 tốt – 5 kém)",
    "MentHlth": "Số ngày sức khỏe tinh thần không tốt (30 ngày qua)",
    "PhysHlth": "Số ngày sức khỏe thể chất không tốt (30 ngày qua)",
    "DiffWalk": "Khó khăn khi đi bộ hoặc leo cầu thang",
    "Sex": "Giới tính",
    "Age": "Nhóm tuổi (13 nhóm)",
    "Education": "Trình độ học vấn (6 mức)",
    "Income": "Thu nhập hộ gia đình (8 mức)",
}

AGE_LABELS: dict[int, str] = {
    1: "18–24",
    2: "25–29",
    3: "30–34",
    4: "35–39",
    5: "40–44",
    6: "45–49",
    7: "50–54",
    8: "55–59",
    9: "60–64",
    10: "65–69",
    11: "70–74",
    12: "75–79",
    13: "80+",
}

INCOME_LABELS: dict[int, str] = {
    1: "<10k USD",
    2: "10–15k",
    3: "15–20k",
    4: "20–25k",
    5: "25–35k",
    6: "35–50k",
    7: "50–75k",
    8: "≥75k",
}

EDUCATION_LABELS: dict[int, str] = {
    1: "Chưa từng đi học hoặc mẫu giáo",
    2: "Tiểu học (lớp 1–8)",
    3: "THCS hoặc chưa tốt nghiệp THPT (lớp 9–11)",
    4: "Tốt nghiệp THPT hoặc tương đương",
    5: "Cao đẳng hoặc đại học dang dở (1–3 năm)",
    6: "Tốt nghiệp đại học trở lên (≥4 năm)",
}

GENHLTH_LABELS: dict[int, str] = {
    1: "Rất tốt",
    2: "Tốt",
    3: "Khá",
    4: "Trung bình",
    5: "Kém",
}

SEX_LABELS: dict[int, str] = {
    0: "Nữ",
    1: "Nam",
}


def vi(name: str) -> str:
    """Trả về tên tiếng Việt của đặc trưng, nếu không có khóa trả lại tên gốc.

    Args:
        name: Tên tiếng Anh của đặc trưng.

    Returns:
        str: Tên tiếng Việt hoặc name nếu không tìm thấy.
    """
    return VI_FEATURE.get(name, name)
