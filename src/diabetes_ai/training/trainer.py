"""Vòng lặp huấn luyện MLP và chọn epoch tốt nhất theo AUC validation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation.metrics import macro_f1, roc_auc_ovr
from ..logging_utils import get_logger
from ..seed import set_seed
from .callbacks import BestEpochTracker, EarlyStopping

logger = get_logger(__name__)


class Trainer:
    """Quản lý huấn luyện MLP, theo dõi metrics và khôi phục trọng số epoch tốt nhất."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 1024,
        epochs: int = 14,
        seed: int = 42,
        device: str = "cpu",
        patience: int | None = None,
    ) -> None:
        """Khởi tạo Trainer.

        Args:
            model: Mô hình nn.Module (MLPNet).
            loss_fn: Hàm mất mát (CrossEntropyLoss có trọng số hoặc không).
            lr: Tốc độ học (learning rate).
            weight_decay: Hệ số suy giảm trọng số L2.
            batch_size: Kích thước lô.
            epochs: Số epoch tối đa.
            seed: Seed số ngẫu nhiên để tái lập.
            device: Thiết bị chạy ('cpu').
            patience: Số epoch kiên nhẫn cho EarlyStopping (None nếu không dừng sớm).
        """
        self.model: nn.Module = model
        self.loss_fn: nn.Module = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.lr: float = float(lr)
        self.weight_decay: float = float(weight_decay)
        self.batch_size: int = int(batch_size)
        self.epochs: int = int(epochs)
        self.seed: int = int(seed)
        self.device: torch.device = torch.device(device)
        self.patience: int | None = patience

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Thực thi vòng lặp huấn luyện từng epoch.

        Args:
            X_train: Dữ liệu đặc trưng huấn luyện.
            y_train: Nhãn tập huấn luyện.
            X_val: Dữ liệu đặc trưng validation.
            y_val: Nhãn tập validation.

        Returns:
            dict chứa history (danh sách từng epoch), best_epoch và best_value.
        """
        set_seed(self.seed)
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        X_tr = torch.as_tensor(X_train, dtype=torch.float32)
        y_tr = torch.as_tensor(y_train, dtype=torch.long)
        train_dataset = TensorDataset(X_tr, y_tr)

        g = torch.Generator()
        g.manual_seed(self.seed)
        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=g,
            num_workers=0,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        tracker = BestEpochTracker(mode="max")
        early_stopping = EarlyStopping(patience=self.patience)

        history: list[dict[str, Any]] = []

        val_x_np = X_train if X_val is None else X_val
        val_y_np = y_train if y_val is None else y_val
        val_y_arr = np.asarray(val_y_np, dtype=np.int64)

        Xv = torch.as_tensor(val_x_np, dtype=torch.float32, device=self.device)
        yv = torch.as_tensor(val_y_arr, dtype=torch.long, device=self.device)

        for epoch in range(1, self.epochs + 1):
            t_epoch_start = time.perf_counter()

            # 1. Huấn luyện qua DataLoader
            self.model.train()
            total_train_loss = 0.0
            n_samples = 0

            for bx, by in loader:
                bx = bx.to(self.device)
                by = by.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(bx)
                loss = self.loss_fn(logits, by)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * len(by)
                n_samples += len(by)

            train_loss = total_train_loss / max(n_samples, 1)

            # 2. Đánh giá trên Validation (chỉ theo dõi, không cập nhật trọng số)
            self.model.eval()
            with torch.no_grad():
                logits_v = self.model(Xv)
                val_loss = nn.functional.cross_entropy(logits_v, yv).item()
                pv = torch.softmax(logits_v, dim=1).cpu().numpy().astype("float64")

            auc = roc_auc_ovr(val_y_arr, pv)
            y_pred_val = np.argmax(pv, axis=1)
            n_classes = pv.shape[1]
            f1 = macro_f1(val_y_arr, y_pred_val, n_classes=n_classes)
            seconds = float(time.perf_counter() - t_epoch_start)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_auc_ovr": float(auc),
                    "val_macro_f1": float(f1),
                    "seconds": float(seconds),
                }
            )

            # 3. Theo dõi epoch tốt nhất và kiểm tra early stopping
            metric_for_tracker = float(auc) if not np.isnan(auc) else -1.0
            improved = tracker.update(epoch, metric_for_tracker, self.model)
            if early_stopping.step(improved):
                logger.info("Early stopping được kích hoạt tại epoch %d.", epoch)
                break

        # 4. Khôi phục trọng số tốt nhất
        if tracker.best_state is not None:
            self.model.load_state_dict(tracker.best_state)
            logger.info("Đã khôi phục trọng số tốt nhất tại epoch %s", tracker.best_epoch)

        return {
            "history": history,
            "best_epoch": tracker.best_epoch,
            "best_value": tracker.best_value,
        }
