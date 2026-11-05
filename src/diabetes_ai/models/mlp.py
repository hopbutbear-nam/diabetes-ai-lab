"""Mạng nơ-ron đa tầng (MLP) PyTorch và lớp mô hình MLPModel.

Cung cấp:
- MLPNet: Module PyTorch cấu trúc n_in -> hidden -> n_out với Dropout và ReLU.
- MLPModel: Bọc MLPNet tương thích BaseModel, huấn luyện qua Trainer và lưu/nạp model.pt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..logging_utils import get_logger
from .base import BaseModel, ensure_proba

logger = get_logger(__name__)


class MLPNet(nn.Module):
    """Module PyTorch hiện thực kiến trúc MLP cho phân loại đa lớp/nhị phân.

    Cấu trúc mặc định: 21 -> 128 -> 64 -> 32 -> 3 (ReLU, dropout sau 2 lớp ẩn đầu).
    """

    def __init__(
        self,
        n_in: int = 21,
        hidden: Sequence[int] = (128, 64, 32),
        n_out: int = 3,
        dropout: Sequence[float] = (0.15, 0.1),
        activation: str = "relu",
    ) -> None:
        """Khởi tạo kiến trúc MLPNet.

        Args:
            n_in: Số chiều đầu vào (mặc định 21).
            hidden: Chuỗi số nút của các tầng ẩn (mặc định (128, 64, 32)).
            n_out: Số nút đầu ra tương ứng số lớp (mặc định 3).
            dropout: Tỷ lệ dropout áp dụng sau các tầng ẩn đầu (mặc định (0.15, 0.1)).
            activation: Hàm kích hoạt ('relu').

        Raises:
            ValueError: Nếu activation khác 'relu'.
        """
        super().__init__()
        if activation != "relu":
            raise ValueError(f"Chỉ hỗ trợ activation 'relu', nhận được: '{activation}'")

        self.n_in: int = int(n_in)
        self.hidden: tuple[int, ...] = tuple(int(h) for h in hidden)
        self.n_out: int = int(n_out)
        self.dropout: tuple[float, ...] = tuple(float(d) for d in dropout)
        self.activation: str = activation

        layers: list[nn.Module] = []
        d_in = self.n_in
        pads = list(self.dropout) + [0.0] * (len(self.hidden) - len(self.dropout))
        for h, p in zip(self.hidden, pads):
            layers += [nn.Linear(d_in, h), nn.ReLU()] + ([nn.Dropout(p)] if p > 0 else [])
            d_in = h
        layers.append(nn.Linear(d_in, self.n_out))
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi tính logits từ đầu vào x."""
        return self.net(x)

    def export_numpy(self) -> dict[str, Any]:
        """Xuất danh sách ma trận trọng số W và vector bias b của từng lớp Linear."""
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        layers: list[dict[str, np.ndarray]] = []
        for module in self.net:
            if isinstance(module, nn.Linear):
                w = module.weight.detach().cpu().numpy().copy()
                b = (
                    module.bias.detach().cpu().numpy().copy()
                    if module.bias is not None
                    else np.zeros(module.out_features, dtype=np.float32)
                )
                weights.append(w)
                biases.append(b)
                layers.append({"weight": w, "bias": b})

        return {
            "weights": weights,
            "biases": biases,
            "layers": layers,
            "arch": {
                "hidden": list(self.hidden),
                "dropout": list(self.dropout),
                "activation": self.activation,
            },
            "n_in": self.n_in,
            "n_out": self.n_out,
        }


class MLPModel(BaseModel):
    """Mô hình MLP tương thích giao diện chuẩn BaseModel."""

    name: ClassVar[str] = "mlp"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        n_classes: int = 3,
        seed: int = 42,
    ) -> None:
        """Khởi tạo MLPModel."""
        super().__init__(params=params, n_classes=n_classes, seed=seed)

        arch_cfg = self.params.get("arch", {})
        if not isinstance(arch_cfg, dict):
            arch_cfg = {}

        self.arch: dict[str, Any] = {
            "hidden": list(arch_cfg.get("hidden", self.params.get("hidden", [128, 64, 32]))),
            "dropout": list(arch_cfg.get("dropout", self.params.get("dropout", [0.15, 0.1]))),
            "activation": str(arch_cfg.get("activation", self.params.get("activation", "relu"))),
        }
        self.n_in: int = int(self.params.get("n_in", 21))

        train_cfg = self.params.get("train", {})
        if not isinstance(train_cfg, dict):
            train_cfg = {}
        self.device: str = str(train_cfg.get("device", self.params.get("device", "cpu")))

        torch.manual_seed(self.seed)
        self.net: MLPNet = MLPNet(
            n_in=self.n_in,
            hidden=self.arch["hidden"],
            n_out=self.n_classes,
            dropout=self.arch["dropout"],
            activation=self.arch["activation"],
        )
        self.net.to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> MLPModel:
        """Huấn luyện MLP qua Trainer và lưu fit_info."""
        from ..training.losses import build_loss
        from ..training.trainer import Trainer

        if self.net.n_in != X_train.shape[1]:
            self.n_in = int(X_train.shape[1])
            torch.manual_seed(self.seed)
            self.net = MLPNet(
                n_in=self.n_in,
                hidden=self.arch["hidden"],
                n_out=self.n_classes,
                dropout=self.arch["dropout"],
                activation=self.arch["activation"],
            )
            self.net.to(self.device)

        train_cfg = self.params.get("train", {})
        if not isinstance(train_cfg, dict):
            train_cfg = {}

        lr = float(train_cfg.get("lr", self.params.get("lr", 1e-3)))
        weight_decay = float(train_cfg.get("weight_decay", self.params.get("weight_decay", 1e-5)))
        batch_size = int(train_cfg.get("batch_size", self.params.get("batch_size", 1024)))
        epochs = int(train_cfg.get("epochs", self.params.get("epochs", 14)))
        patience = train_cfg.get("patience", self.params.get("patience", None))
        device = str(train_cfg.get("device", self.params.get("device", self.device)))

        loss_fn = self.params.get("loss_fn", None)
        if loss_fn is None:
            loss_kind = str(self.params.get("loss", "weighted_ce"))
            class_weights_arr: np.ndarray | None = None
            if "class_weights" in self.params:
                class_weights_arr = np.asarray(self.params["class_weights"], dtype=np.float32)
            elif "imbalance" in self.params:
                imb = self.params["imbalance"]
                if isinstance(imb, dict) and imb.get("strategy") == "class_weight":
                    power = float(imb.get("power", 0.75))
                    from ..training.imbalance import class_weights

                    class_weights_arr = class_weights(
                        y_train, n_classes=self.n_classes, power=power
                    )

            loss_fn = build_loss(
                kind=loss_kind,
                class_weights=class_weights_arr,
                device=device,
            )

        trainer = Trainer(
            model=self.net,
            loss_fn=loss_fn,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            seed=self.seed,
            device=device,
            patience=patience,
        )

        fit_res = trainer.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        self.fit_info = {
            "history": fit_res["history"],
            "best_epoch": fit_res["best_epoch"],
            "best_value": fit_res["best_value"],
        }
        logger.info(
            "MLPModel fit thành công: best_epoch=%s, best_value=%s",
            self.fit_info["best_epoch"],
            self.fit_info["best_value"],
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán ma trận xác suất theo lô 8192."""
        if self.net is None:
            raise RuntimeError("Mô hình MLPModel chưa được khởi tạo hoặc huấn luyện.")

        self.net.eval()
        X_np = np.asarray(X, dtype=np.float32)
        n_samples = len(X_np)
        batch_size = 8192
        probas: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_x = torch.as_tensor(
                    X_np[i : i + batch_size], dtype=torch.float32, device=self.device
                )
                logits = self.net(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
                probas.append(probs)

        if len(probas) == 0:
            return np.empty((0, self.n_classes), dtype=np.float64)

        all_probas = np.vstack(probas)
        return ensure_proba(all_probas, self.n_classes)

    def save(self, run_dir: Path) -> None:
        """Lưu model.pt vào run_dir."""
        if self.net is None:
            raise RuntimeError("Mô hình MLPModel chưa được khởi tạo để lưu.")

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        pt_path = run_dir / "model.pt"

        cpu_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        payload = {
            "state_dict": cpu_state,
            "arch": self.arch,
            "n_in": self.n_in,
            "n_out": self.n_classes,
        }
        torch.save(payload, pt_path)
        logger.info("Đã lưu MLPModel tại %s", pt_path)

    @classmethod
    def load(cls, run_dir: Path) -> MLPModel:
        """Tải MLPModel từ run_dir/model.pt."""
        run_dir = Path(run_dir)
        pt_path = run_dir / "model.pt"
        if not pt_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại {pt_path}")

        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        arch = payload["arch"]
        n_in = int(payload["n_in"])
        n_out = int(payload["n_out"])
        state_dict = payload["state_dict"]

        instance = cls(
            params={"arch": arch, "n_in": n_in},
            n_classes=n_out,
        )
        instance.n_in = n_in
        instance.arch = arch
        instance.net = MLPNet(
            n_in=n_in,
            hidden=arch["hidden"],
            n_out=n_out,
            dropout=arch["dropout"],
            activation=arch.get("activation", "relu"),
        )
        instance.net.load_state_dict(state_dict)
        instance.net.eval()
        logger.info("Đã nạp MLPModel từ %s", pt_path)
        return instance

    def export_numpy(self) -> dict[str, Any]:
        """Xuất danh sách W, b từng lớp Linear (dùng ở K08)."""
        if self.net is None:
            raise RuntimeError("Mô hình MLPModel chưa có mạng nơ-ron để xuất trọng số.")
        return self.net.export_numpy()
