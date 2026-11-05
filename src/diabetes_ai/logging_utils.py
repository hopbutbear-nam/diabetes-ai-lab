"""Cấu hình logging chung cho toàn bộ dự án."""

import logging
from pathlib import Path
import sys


def get_logger(
    name: str = "diabetes_ai",
    level: str | None = None,
    log_file: Path | None = None,
) -> logging.Logger:
    """Khởi tạo hoặc lấy logger với cấu hình chuẩn.

    Args:
        name: Tên của logger (mặc định 'diabetes_ai').
        level: Mức log (DEBUG, INFO, WARNING, ERROR, CRITICAL). Mặc định INFO.
        log_file: Đường dẫn tệp ghi log nếu muốn ghi ra file.

    Returns:
        logging.Logger: Đối tượng logger đã cấu hình.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logger = logging.getLogger(name)
    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Tránh thêm lặp lại console handler
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(log_level)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        target_path = str(log_file.resolve())
        has_file_handler = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == target_path
            for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(target_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

    return logger
