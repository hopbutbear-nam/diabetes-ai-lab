"""Early stopping, checkpoint."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BestEpochTracker:
    """Theo dõi epoch tốt nhất dựa trên giá trị metric (AUC, loss, v.v.)."""

    def __init__(self, mode: str = "max") -> None:
        """Khởi tạo tracker với mode 'max' (mặc định) hoặc 'min'."""
        if mode not in ("max", "min"):
            raise ValueError(f"Chế độ mode không hợp lệ: '{mode}'. Chỉ hỗ trợ 'max' hoặc 'min'.")
        self.mode: str = mode
        self.best_epoch: int | None = None
        self.best_value: float | None = None
        self.best_state: dict[str, torch.Tensor] | None = None

    def update(self, epoch: int, value: float, model: nn.Module) -> bool:
        """Cập nhật epoch hiện tại và lưu state_dict tốt nhất về CPU nếu cải thiện."""
        improved = False
        if self.best_value is None:
            improved = True
        elif self.mode == "max":
            improved = value > self.best_value
        elif self.mode == "min":
            improved = value < self.best_value

        if improved:
            self.best_epoch = epoch
            self.best_value = float(value)
            cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_state = copy.deepcopy(cpu_state)
            return True

        return False


class EarlyStopping:
    """Cơ chế dừng sớm (Early Stopping) khi metric không cải thiện."""

    def __init__(self, patience: int | None = None) -> None:
        """Khởi tạo EarlyStopping với số epoch kiên nhẫn (patience). None nếu không dừng sớm."""
        self.patience: int | None = patience
        self.counter: int = 0

    def step(self, improved: bool) -> bool:
        """Ghi nhận kết quả của epoch và trả về True nếu cần dừng huấn luyện."""
        if self.patience is None:
            return False
        if improved:
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(
    path: Path | str,
    state: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> None:
    """Lưu checkpoint mô hình kèm siêu dữ liệu ra file bằng torch.save."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": state,
        "state": state,
        "meta": meta if meta is not None else {},
    }
    torch.save(payload, p)
