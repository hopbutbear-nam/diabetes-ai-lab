"""Cấu hình chung và các fixture cho pytest."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diabetes_ai.config import load_config
from diabetes_ai.data.schema import FEATURES, TARGET
from diabetes_ai.paths import ROOT


@pytest.fixture
def sample_profile():
    """Fixture trả về một mẫu hồ sơ bệnh nhân dạng dict."""
    p = Path(__file__).parent / "fixtures" / "sample_profile.json"
    return json.loads(p.read_text(encoding="utf-8"))


def make_synthetic_df(
    n: int = 4000,
    seed: int = 0,
    dup_frac: float = 0.08,
) -> pd.DataFrame:
    """Tạo DataFrame giả lập đủ 22 cột theo schema BRFSS 2015.

    Args:
        n: Tổng số dòng mong muốn.
        seed: Seed sinh số ngẫu nhiên.
        dup_frac: Tỉ lệ bản sao dòng ngẫu nhiên.

    Returns:
        pd.DataFrame: DataFrame giả lập đủ 22 cột đúng miền giá trị và kiểu dữ liệu.
    """
    rng = np.random.default_rng(seed)

    n_dup = int(round(n * dup_frac)) if dup_frac > 0 else 0
    n_unique = max(1, n - n_dup) if n >= n_dup else n

    # 14 đặc trưng nhị phân Bernoulli (0.0 hoặc 1.0)
    high_bp = rng.binomial(1, 0.4, size=n_unique).astype("float32")
    high_chol = rng.binomial(1, 0.4, size=n_unique).astype("float32")
    chol_check = rng.binomial(1, 0.95, size=n_unique).astype("float32")
    smoker = rng.binomial(1, 0.45, size=n_unique).astype("float32")
    stroke = rng.binomial(1, 0.05, size=n_unique).astype("float32")
    heart_disease = rng.binomial(1, 0.1, size=n_unique).astype("float32")
    phys_activity = rng.binomial(1, 0.75, size=n_unique).astype("float32")
    fruits = rng.binomial(1, 0.6, size=n_unique).astype("float32")
    veggies = rng.binomial(1, 0.8, size=n_unique).astype("float32")
    hvy_alcohol = rng.binomial(1, 0.05, size=n_unique).astype("float32")
    any_healthcare = rng.binomial(1, 0.95, size=n_unique).astype("float32")
    no_docbc_cost = rng.binomial(1, 0.08, size=n_unique).astype("float32")
    diff_walk = rng.binomial(1, 0.15, size=n_unique).astype("float32")
    sex = rng.binomial(1, 0.5, size=n_unique).astype("float32")

    # BMI nguyên ~ N(28, 6) cắt [12, 98]
    bmi_raw = np.round(rng.normal(28, 6, size=n_unique))
    bmi = np.clip(bmi_raw, 12, 98).astype("float32")

    # Đặc trưng rời rạc
    gen_hlth = rng.choice([1, 2, 3, 4, 5], size=n_unique, p=[0.2, 0.35, 0.3, 0.1, 0.05]).astype("float32")
    age = rng.integers(1, 14, size=n_unique).astype("float32")
    education = rng.choice([1, 2, 3, 4, 5, 6], size=n_unique, p=[0.02, 0.03, 0.1, 0.25, 0.3, 0.3]).astype("float32")
    income = rng.integers(1, 9, size=n_unique).astype("float32")

    # MentHlth/PhysHlth (0-30), phần lớn là 0
    ment_hlth = np.where(rng.random(n_unique) < 0.8, 0, rng.integers(1, 31, size=n_unique)).astype("float32")
    phys_hlth = np.where(rng.random(n_unique) < 0.8, 0, rng.integers(1, 31, size=n_unique)).astype("float32")

    # Nhãn sinh từ mô hình logistic kết hợp HighBP, BMI, Age, GenHlth (AUC > 0.7, tỉ lệ ~84/2/14)
    z = -3.5 + 1.2 * high_bp + 0.08 * (bmi - 25) + 0.2 * (age - 7) + 0.5 * (gen_hlth - 2)
    noise = rng.normal(0, 1.0, size=n_unique)
    latent = z + noise
    q84, q86 = np.percentile(latent, [84, 86])
    target = np.zeros(n_unique, dtype=np.int64)
    target[latent >= q84] = 1
    target[latent >= q86] = 2

    data = {
        TARGET: target,
        "HighBP": high_bp,
        "HighChol": high_chol,
        "CholCheck": chol_check,
        "BMI": bmi,
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heart_disease,
        "PhysActivity": phys_activity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": hvy_alcohol,
        "AnyHealthcare": any_healthcare,
        "NoDocbcCost": no_docbc_cost,
        "GenHlth": gen_hlth,
        "MentHlth": ment_hlth,
        "PhysHlth": phys_hlth,
        "DiffWalk": diff_walk,
        "Sex": sex,
        "Age": age,
        "Education": education,
        "Income": income,
    }
    df = pd.DataFrame(data)

    # Thêm các dòng trùng ngẫu nhiên
    if n_dup > 0 and len(df) > 0:
        dup_indices = rng.choice(len(df), size=n_dup, replace=True)
        dup_df = df.iloc[dup_indices]
        df = pd.concat([df, dup_df], ignore_index=True)

    df[TARGET] = df[TARGET].astype("int64")
    for col in FEATURES:
        df[col] = df[col].astype("float32")

    return df.reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Fixture cung cấp DataFrame giả lập dùng cho toàn bộ phiên test."""
    return make_synthetic_df()


@pytest.fixture
def real_csv_path() -> Path:
    """Fixture trả về đường dẫn CSV dữ liệu thật hoặc skip nếu file chưa có."""
    try:
        cfg = load_config("data")
        raw_path = cfg.get("raw_path", "data/raw/brfss2015/diabetes_012_health_indicators_BRFSS2015.csv")
        csv_file = ROOT / raw_path
    except Exception:
        csv_file = ROOT / "data" / "raw" / "brfss2015" / "diabetes_012_health_indicators_BRFSS2015.csv"

    if not csv_file.is_file():
        pytest.skip("Không có CSV thật")

    return csv_file
