"""Nạp YAML trong configs/, ghép với biến môi trường (.env)."""

import copy
import os
from pathlib import Path
from typing import Any

import dotenv
import yaml

from diabetes_ai.paths import CONFIGS_DIR, ROOT

_DOTENV_LOADED = False


def load_yaml(path: str | Path) -> dict:
    """Đọc file YAML với mã hóa utf-8.

    Args:
        path: Đường dẫn tới file YAML.

    Returns:
        dict: Dữ liệu cấu hình đã nạp.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
    """
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_config(name: str) -> dict:
    """Nạp cấu hình từ thư mục configs/ theo tên file.

    Args:
        name: Tên cấu hình (ví dụ 'data' hoặc 'data.yaml').

    Returns:
        dict: Dữ liệu cấu hình.
    """
    filename = name if name.endswith(".yaml") or name.endswith(".yml") else f"{name}.yaml"
    return load_yaml(CONFIGS_DIR / filename)


def deep_update(base: dict, override: dict) -> dict:
    """Cập nhật đệ quy từ điển override vào base mà không sửa đối tượng gốc."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_env(key: str, default: str | None = None) -> str | None:
    """Lấy biến môi trường từ hệ thống hoặc file .env (nạp đúng một lần)."""
    global _DOTENV_LOADED
    if not _DOTENV_LOADED:
        dotenv.load_dotenv(ROOT / ".env")
        _DOTENV_LOADED = True
    return os.environ.get(key, default)
