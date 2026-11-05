"""Tests cho BestEpochTracker, EarlyStopping và Trainer.fit với dữ liệu giả lập."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from diabetes_ai.training.callbacks import BestEpochTracker, EarlyStopping
from diabetes_ai.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Test 1: BestEpochTracker chọn đúng epoch tốt nhất
# ---------------------------------------------------------------------------


def test_best_epoch_tracker_max() -> None:
    """Chuỗi AUC 0.6 → 0.7 → 0.65: epoch tốt nhất phải là 2 (AUC = 0.7)."""
    model = nn.Linear(2, 2)
    tracker = BestEpochTracker(mode="max")

    tracker.update(1, 0.6, model)
    tracker.update(2, 0.7, model)
    tracker.update(3, 0.65, model)

    assert tracker.best_epoch == 2
    assert abs(tracker.best_value - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# Test 2: EarlyStopping dừng đúng sau patience lần không cải thiện
# ---------------------------------------------------------------------------


def test_early_stopping_patience() -> None:
    """patience=2: dừng sau đúng 2 lần không cải thiện liên tiếp."""
    es = EarlyStopping(patience=2)

    # Epoch 1 cải thiện → chưa dừng
    assert es.step(improved=True) is False
    # Epoch 2 không cải thiện (counter=1) → chưa dừng
    assert es.step(improved=False) is False
    # Epoch 3 không cải thiện (counter=2) → phải dừng
    assert es.step(improved=False) is True


# ---------------------------------------------------------------------------
# Test 3: Trainer.fit — 3 epoch, khôi phục trọng số epoch tốt nhất
# ---------------------------------------------------------------------------


class _TinyNet(nn.Module):
    """Mạng tuyến tính nhỏ cho dữ liệu giả lập 3 lớp."""

    def __init__(self, n_in: int = 4, n_out: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_trainer_fit_restores_best() -> None:
    """3 epoch trên dữ liệu giả lập:
    - len(history) == 3
    - val_auc_ovr của best_epoch là lớn nhất trong tất cả epoch
    - trọng số sau fit cho AUC đó ≤ 1e-9 sai lệch
    """
    rng = np.random.default_rng(42)
    n_samples, n_in, n_classes = 300, 4, 3

    X = rng.standard_normal((n_samples, n_in)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_samples).astype(np.int64)

    X_val = rng.standard_normal((60, n_in)).astype(np.float32)
    y_val = rng.integers(0, n_classes, size=60).astype(np.int64)

    net = _TinyNet(n_in=n_in, n_out=n_classes)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=net,
        loss_fn=loss_fn,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=64,
        epochs=3,
        seed=42,
        device="cpu",
        patience=None,
    )

    result = trainer.fit(X_train=X, y_train=y, X_val=X_val, y_val=y_val)
    history = result["history"]
    best_epoch = result["best_epoch"]

    # Kiểm tra: đủ 3 epoch
    assert len(history) == 3

    # Kiểm tra: best_epoch có val_auc_ovr cao nhất
    aucs = [row["val_auc_ovr"] for row in history]
    best_auc_in_history = history[best_epoch - 1]["val_auc_ovr"]
    assert best_auc_in_history == max(aucs)

    # Kiểm tra: trọng số sau fit cho đúng AUC tương ứng (≤ 1e-9 sai lệch)
    from sklearn.metrics import roc_auc_score

    net.eval()
    with torch.no_grad():
        logits = net(torch.as_tensor(X_val, dtype=torch.float32))
        proba = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)

    # Tính AUC thực tế từ trọng số đã được khôi phục
    auc_restored = roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
    assert abs(auc_restored - best_auc_in_history) <= 1e-9
