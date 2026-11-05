"""Lược đồ dữ liệu BRFSS 2015 (21 đặc trưng) và nhãn."""

TARGET = "Diabetes_012"
CLASSES = ["không", "tiền tiểu đường", "tiểu đường"]
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]

ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]

COUNT_FEATURES = ["MentHlth", "PhysHlth"]

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    # 14 đặc trưng nhị phân (0, 1)
    "HighBP": (0.0, 1.0),
    "HighChol": (0.0, 1.0),
    "CholCheck": (0.0, 1.0),
    "Smoker": (0.0, 1.0),
    "Stroke": (0.0, 1.0),
    "HeartDiseaseorAttack": (0.0, 1.0),
    "PhysActivity": (0.0, 1.0),
    "Fruits": (0.0, 1.0),
    "Veggies": (0.0, 1.0),
    "HvyAlcoholConsump": (0.0, 1.0),
    "AnyHealthcare": (0.0, 1.0),
    "NoDocbcCost": (0.0, 1.0),
    "DiffWalk": (0.0, 1.0),
    "Sex": (0.0, 1.0),
    # Các đặc trưng liên tục, đếm, thứ bậc
    "BMI": (10.0, 100.0),
    "MentHlth": (0.0, 30.0),
    "PhysHlth": (0.0, 30.0),
    "GenHlth": (1.0, 5.0),
    "Age": (1.0, 13.0),
    "Education": (1.0, 6.0),
    "Income": (1.0, 8.0),
}

BINARY_CLASS_NAMES = ["không", "tiền tiểu đường hoặc tiểu đường"]


def class_names(task: str) -> list[str]:
    """Trả về danh sách tên lớp tương ứng với tác vụ.

    Args:
        task: Tác vụ phân loại ('multiclass' hoặc 'binary').

    Returns:
        list[str]: Danh sách tên các lớp.

    Raises:
        ValueError: Nếu task không hợp lệ.
    """
    if task == "multiclass":
        return CLASSES
    elif task == "binary":
        return BINARY_CLASS_NAMES
    else:
        raise ValueError(
            f"Tác vụ không hợp lệ: '{task}'. Chỉ chấp nhận 'multiclass' hoặc 'binary'."
        )
