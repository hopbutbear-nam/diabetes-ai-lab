"""Lược đồ dữ liệu BRFSS 2015 (21 đặc trưng) và nhãn."""
TARGET = "Diabetes_012"
CLASSES = ["không", "tiền tiểu đường", "tiểu đường"]
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
