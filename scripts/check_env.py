#!/usr/bin/env python3
"""Kiểm tra môi trường thực thi và tính toàn vẹn dữ liệu dự án diabetes-ai."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


def _ver(pkg: str) -> str | None:
    """Lấy phiên bản gói cài đặt qua importlib.metadata."""
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return None


def _try_import(mod: str) -> tuple[bool, str]:
    """Thử import module và trả về (thành công, thông điệp lỗi)."""
    try:
        importlib.import_module(mod)
        return True, ""
    except Exception as e:
        return False, str(e)


def _package_versions() -> dict[str, str | None]:
    """Phiên bản các gói chính (None nếu chưa cài)."""
    from importlib import metadata

    names = ["numpy", "pandas", "scikit-learn", "xgboost", "torch", "shap", "fastapi", "pydantic", "uvicorn", "joblib", "pyyaml"]
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    repo_root = Path(__file__).resolve().parents[1]

    # Kiểm tra venv & thông tin hệ thống
    in_venv = sys.prefix != sys.base_prefix
    py_ver_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    plat_str = platform.platform()

    print("=" * 68)
    print("                 KIỂM TRA MÔI TRƯỜNG THỰC TẾ")
    print("=" * 68)
    print(f"Python: {py_ver_str} ({sys.executable})")
    print(f"Nền tảng: {plat_str}")
    print(f"Trong môi trường ảo (.venv): {in_venv}")

    if sys.version_info >= (3, 14):
        print("[CẢNH BÁO] Python >= 3.14: Một số thư viện C-extension/wheel có thể chưa hoàn toàn tương thích.")

    repo_str = str(repo_root)
    if any(c.isspace() or ord(c) > 127 for c in repo_str):
        print(f"[THÔNG TIN] Đường dẫn thư mục chứa khoảng trắng hoặc ký tự không phải ASCII: {repo_str}")

    # Thử import diabetes_ai
    diabetes_ai_path: str | None = None
    try:
        if str(repo_root / "src") not in sys.path:
            sys.path.insert(0, str(repo_root / "src"))
        import diabetes_ai  # type: ignore

        diabetes_ai_file = getattr(diabetes_ai, "__file__", None)
        if diabetes_ai_file:
            diabetes_ai_path = str(Path(diabetes_ai_file).resolve())
            print(f"diabetes_ai: {diabetes_ai_path}")
            src_path = str((repo_root / "src").resolve())
            if not diabetes_ai_path.startswith(src_path):
                print(f"[CẢNH BÁO] diabetes_ai không nằm dưới src/: {diabetes_ai_path}")
    except Exception as e:
        print(f"[CẢNH BÁO] Chưa thể import diabetes_ai: {e}")

    # Danh sách gói lõi CORE
    core_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("pyyaml", "yaml"),
        ("joblib", "joblib"),
        ("matplotlib", "matplotlib"),
        ("fastapi", "fastapi"),
        ("pydantic", "pydantic"),
        ("uvicorn", "uvicorn"),
        ("pytest", "pytest"),
        ("httpx", "httpx"),
    ]

    print("\n" + "-" * 68)
    print(f"{'Gói lõi (CORE)':<20} | {'Phiên bản':<16} | {'Trạng thái'}")
    print("-" * 68)

    core_results = {}
    core_ok = True
    for pkg_name, mod_name in core_packages:
        ver = _ver(pkg_name)
        ok, err = _try_import(mod_name)
        status = "OK" if ok else f"LỖI: {err}"
        if not ok:
            core_ok = False
        core_results[pkg_name] = {"version": ver, "ok": ok, "error": err}
        print(f"{pkg_name:<20} | {str(ver):<16} | {status}")

    # Smoke test thư viện nặng HEAVY
    print("\n" + "-" * 68)
    print("Smoke test thư viện nặng (HEAVY):")
    print("-" * 68)

    # 1. torch
    torch_ok = False
    cuda_available = False
    torch_ver = _ver("torch")
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        x = torch.randn(2, 2)
        _ = x @ x
        torch_ok = True
        print(f"torch: {torch_ver} (CUDA={cuda_available}, smoke @ OK)")
    except Exception as e:
        print(f"torch: LỖI smoke test: {e}")

    # 2. xgboost
    xgboost_ok = False
    xgboost_ver = _ver("xgboost")
    try:
        import numpy as np
        import xgboost as xgb

        m = xgb.XGBClassifier(n_estimators=5, random_state=42, eval_metric="logloss")
        rng = np.random.default_rng(42)
        X_dummy = rng.normal(size=(200, 4))
        y_dummy = rng.integers(0, 2, size=200)
        m.fit(X_dummy, y_dummy)
        xgboost_ok = True
        print(f"xgboost: {xgboost_ver} (fit 5 cây trên 200 dòng OK)")
    except Exception as e:
        print(f"xgboost: LỖI smoke test: {e}")

    # 3. shap
    shap_ok = False
    shap_ver = _ver("shap")
    try:
        import shap  # noqa: F401

        shap_ok = True
        print(f"shap: {shap_ver} (import OK)")
    except Exception as e:
        print(f"shap: LỖI import: {e}")

    # Kiểm tra CSV dữ liệu
    csv_path = repo_root / "data" / "raw" / "brfss2015" / "diabetes_012_health_indicators_BRFSS2015.csv"
    csv_exists = csv_path.exists()
    csv_bytes = 0
    csv_sha256: str | None = None
    csv_n_rows: int | None = None

    print("\n" + "-" * 68)
    print("Kiểm tra tệp dữ liệu CSV:")
    print("-" * 68)
    print(f"Đường dẫn: {csv_path}")
    print(f"Tồn tại: {csv_exists}")

    if csv_exists:
        csv_bytes = csv_path.stat().st_size
        h = hashlib.sha256()
        with open(csv_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        csv_sha256 = h.hexdigest()

        with open(csv_path, "r", encoding="utf-8") as f:
            csv_n_rows = sum(1 for _ in f) - 1

        print(f"Kích thước: {csv_bytes} bytes")
        print(f"SHA-256: {csv_sha256}")
        print(f"Số dòng dữ liệu: {csv_n_rows}")
    else:
        print("[LỖI] Không tìm thấy tệp CSV tại đường dẫn trên.")

    # Hành động kiến nghị
    recommended_actions: list[str] = []
    if not in_venv:
        recommended_actions.append(
            "Kích hoạt môi trường ảo: source .venv/Scripts/activate (Bash) hoặc .venv\\Scripts\\Activate.ps1 (PowerShell)"
        )
    if not core_ok:
        recommended_actions.append("Cài đặt các gói lõi: pip install -r requirements-dev.txt && pip install -e .")
    if not torch_ok:
        recommended_actions.append("Cài đặt torch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    if not xgboost_ok:
        recommended_actions.append("Cài đặt xgboost: pip install xgboost")
    if not csv_exists:
        recommended_actions.append(
            "Tải dữ liệu BRFSS2015 vào data/raw/brfss2015/diabetes_012_health_indicators_BRFSS2015.csv"
        )

    # Ghi file reports/metrics/env_check.json
    metrics_dir = repo_root / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_file = metrics_dir / "env_check.json"

    report_payload = {
        "python": py_ver_str,
        "platform": plat_str,
        "in_venv": in_venv,
        "core_ok": core_ok,
        "torch_ok": torch_ok,
        "xgboost_ok": xgboost_ok,
        "shap_ok": shap_ok,
        "cuda": cuda_available,
        "packages": _package_versions(),
        "csv": {
            "exists": csv_exists,
            "bytes": csv_bytes,
            "sha256": csv_sha256,
            "n_rows": csv_n_rows,
        },
        "recommended_actions": recommended_actions,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 68)
    print(f"Đã ghi báo cáo: {report_file}")
    print(f"Kết quả tổng quát: core_ok={core_ok}, torch_ok={torch_ok}, csv_exists={csv_exists}")
    print("=" * 68)

    # Thoát: 0 khi core_ok và csv.exists, ngược lại 1
    if core_ok and csv_exists:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
