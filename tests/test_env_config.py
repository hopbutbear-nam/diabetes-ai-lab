"""Kiểm tra get_env nạp được biến từ file .env."""

import os

import diabetes_ai.config as cfg


def test_get_env_reads_dotenv(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("MODEL_PATH=models/from_dotenv.joblib\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    monkeypatch.setattr(cfg, "_DOTENV_LOADED", False)
    os.environ.pop("MODEL_PATH", None)
    try:
        assert cfg.get_env("MODEL_PATH") == "models/from_dotenv.joblib"
    finally:
        os.environ.pop("MODEL_PATH", None)  # load_dotenv ghi vào os.environ, phải dọn
