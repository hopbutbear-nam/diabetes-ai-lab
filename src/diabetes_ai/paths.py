"""Hằng số đường dẫn: ROOT, DATA_DIR, MODELS_DIR, REPORTS_DIR."""

import os
from pathlib import Path

ROOT: Path = Path(os.environ.get("DIABETES_AI_ROOT") or Path(__file__).resolve().parents[2])

CONFIGS_DIR: Path = ROOT / "configs"
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw" / "brfss2015"
SPLITS_DIR: Path = DATA_DIR / "splits"
MODELS_DIR: Path = ROOT / "models"
REPORTS_DIR: Path = ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"
DOCS_DIR: Path = ROOT / "docs"

ALL_DIRS: list[Path] = [
    CONFIGS_DIR,
    DATA_DIR,
    RAW_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    DOCS_DIR,
]


def ensure_dirs() -> None:
    """Tạo tất cả các thư mục cần thiết trong dự án nếu chưa tồn tại."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def rel(p: Path | str) -> str:
    """Trả về đường dẫn tương đối dạng POSIX so với ROOT để ghi vào JSON/log."""
    path_obj = Path(p)
    try:
        return path_obj.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path_obj.as_posix()
