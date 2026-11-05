"""Bộ chuẩn hóa chuẩn tắc (StandardScaler) sử dụng NumPy."""

import json
from pathlib import Path
import numpy as np


class StandardScalerNP:
    """Chuẩn hóa dữ liệu theo trung bình và độ lệch chuẩn bằng NumPy."""

    def __init__(self, feature_names: list[str] | None = None) -> None:
        """Khởi tạo StandardScalerNP.

        Args:
            feature_names: Danh sách tên cột tương ứng (tùy chọn).
        """
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerNP":
        """Học trung bình và độ lệch chuẩn từ dữ liệu.

        Args:
            X: Ma trận đặc trưng 2D.

        Returns:
            self: Đối tượng đã được fit.

        Raises:
            ValueError: Nếu số chiều không hợp lệ hoặc không khớp feature_names.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Dữ liệu đầu vào phải là mảng 2 chiều, nhận {X_arr.ndim} chiều")

        if self.feature_names is not None and len(self.feature_names) != X_arr.shape[1]:
            raise ValueError(
                f"Cần {len(self.feature_names)} cột theo feature_names, nhận {X_arr.shape[1]}"
            )

        mean = np.mean(X_arr, axis=0, dtype=np.float64)
        std = np.std(X_arr, axis=0, dtype=np.float64, ddof=0)
        # Tránh chia cho 0 khi đặc trưng hằng số
        std = np.where(std < 1e-8, 1.0, std)

        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Chuẩn hóa ma trận đặc trưng thành float32.

        Args:
            X: Ma trận đặc trưng 2D.

        Returns:
            np.ndarray: Ma trận đã chuẩn hóa kiểu float32.

        Raises:
            RuntimeError: Nếu scaler chưa được fit.
            ValueError: Nếu số cột không khớp.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler chưa được fit")

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"Dữ liệu đầu vào phải là mảng 2 chiều, nhận {X_arr.ndim} chiều")

        d = len(self.mean_)
        if X_arr.shape[1] != d:
            raise ValueError(f"Cần {d} cột, nhận {X_arr.shape[1]}")

        return ((X_arr - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Học thông số và chuẩn hóa dữ liệu.

        Args:
            X: Ma trận đặc trưng 2D.

        Returns:
            np.ndarray: Ma trận đã chuẩn hóa kiểu float32.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Đưa ma trận đặc trưng chuẩn hóa về miền giá trị ban đầu.

        Args:
            X: Ma trận đặc trưng đã chuẩn hóa.

        Returns:
            np.ndarray: Ma trận giá trị ban đầu kiểu float32.

        Raises:
            RuntimeError: Nếu scaler chưa được fit.
            ValueError: Nếu số cột không khớp.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler chưa được fit")

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"Dữ liệu đầu vào phải là mảng 2 chiều, nhận {X_arr.ndim} chiều")

        d = len(self.mean_)
        if X_arr.shape[1] != d:
            raise ValueError(f"Cần {d} cột, nhận {X_arr.shape[1]}")

        return (X_arr * self.std_ + self.mean_).astype(np.float32)

    def to_dict(self) -> dict:
        """Chuyển đổi trạng thái scaler thành dictionary.

        Returns:
            dict: Trạng thái của scaler.

        Raises:
            RuntimeError: Nếu scaler chưa được fit.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler chưa được fit")
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StandardScalerNP":
        """Tái tạo scaler từ dictionary.

        Args:
            data: Dictionary chứa thông số scaler.

        Returns:
            StandardScalerNP: Đối tượng scaler đã khôi phục trạng thái.
        """
        scaler = cls(feature_names=data.get("feature_names"))
        mean_val = data.get("mean", data.get("mean_"))
        std_val = data.get("std", data.get("std_"))
        if mean_val is not None:
            scaler.mean_ = np.array(mean_val, dtype=np.float64)
        if std_val is not None:
            scaler.std_ = np.array(std_val, dtype=np.float64)
        return scaler

    def save_json(self, path: Path | str) -> None:
        """Lưu trạng thái scaler ra file JSON.

        Args:
            path: Đường dẫn tệp JSON cần lưu.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> "StandardScalerNP":
        """Nạp scaler từ file JSON.

        Args:
            path: Đường dẫn tệp JSON cần nạp.

        Returns:
            StandardScalerNP: Đối tượng scaler đã nạp thông số.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
