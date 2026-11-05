"""PyTorch Dataset/DataLoader cho MLP."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    """PyTorch Dataset cho dữ liệu dạng bảng (tabular)."""

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """Khởi tạo TabularDataset với X (float32) và y (int64 nếu có)."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        self.X: torch.Tensor = torch.as_tensor(X, dtype=torch.float32)

        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.asarray(y)
            self.y: torch.Tensor | None = torch.as_tensor(y, dtype=torch.int64)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def make_loader(
    X: np.ndarray,
    y: np.ndarray | None = None,
    batch_size: int = 1024,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Tạo DataLoader PyTorch với torch.Generator seed xác định và drop_last=False."""
    dataset = TabularDataset(X, y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
        num_workers=num_workers,
    )
