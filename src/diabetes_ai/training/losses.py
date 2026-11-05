"""Cross-entropy có trọng số lớp (mũ 0.75), focal loss (tùy chọn)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss cho bài toán phân loại đa lớp hoặc nhị phân."""

    def __init__(
        self,
        weight: torch.Tensor | np.ndarray | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma: float = float(gamma)
        self.reduction: str = reduction

        weight_tensor: torch.Tensor | None = None
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight_tensor = torch.as_tensor(weight, dtype=torch.float32)
            else:
                weight_tensor = weight.to(dtype=torch.float32)

        self.register_buffer("weight", weight_tensor)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Tính Focal Loss giữa logits inputs (N, C) và targets nhãn lớp (N,)."""
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        focal_weight = torch.pow(1.0 - target_probs, self.gamma)
        loss = -focal_weight * target_log_probs

        if self.weight is not None:
            class_weights = self.weight[targets]
            loss = loss * class_weights

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    kind: str = "weighted_ce",
    class_weights: np.ndarray | None = None,
    gamma: float = 2.0,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Xây dựng loss function dựa theo loại chỉ định ('weighted_ce', 'focal')."""
    weight_tensor: torch.Tensor | None = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    if kind in ("weighted_ce", "ce"):
        loss_fn: nn.Module = nn.CrossEntropyLoss(weight=weight_tensor)
    elif kind == "focal":
        loss_fn = FocalLoss(weight=weight_tensor, gamma=gamma)
    else:
        raise ValueError(
            f"Loại loss không hợp lệ: '{kind}'. Chỉ hỗ trợ 'weighted_ce' hoặc 'focal'."
        )

    return loss_fn.to(device)
