"""Đặt seed cho random, numpy, torch để tái lập kết quả."""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Đặt seed cho PYTHONHASHSEED, random, numpy và torch để tái lập kết quả."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def make_rng(seed: int = 42) -> np.random.Generator:
    """Tạo bộ sinh số ngẫu nhiên Generator của numpy từ seed."""
    return np.random.default_rng(seed)
