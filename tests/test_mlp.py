"""4 test cho MLP PyTorch (K04b): shape/sum, tính xác định, lưu/nạp và overfit batch nhỏ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from diabetes_ai.models.mlp import MLPModel, MLPNet


def test_forward_shape_and_sum() -> None:
    """Kiểm tra đầu ra forward của mạng có shape (n, 3) và softmax có tổng bằng 1."""
    n_samples = 30
    n_in = 21
    n_out = 3

    x = torch.randn(n_samples, n_in)
    net = MLPNet(n_in=n_in, hidden=(64, 32, 16), n_out=n_out, dropout=(0.15, 0.1))
    net.eval()

    with torch.no_grad():
        out = net(x)
        probs = torch.softmax(out, dim=1).numpy().astype("float64")

    assert out.shape == (n_samples, n_out), f"Shape đầu ra {out.shape} khác {(n_samples, n_out)}"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6), "Tổng softmax mỗi hàng phải bằng 1"

    # Kiểm tra tương tự qua predict_proba của MLPModel
    model = MLPModel(
        params={"hidden": (64, 32, 16), "dropout": (0.15, 0.1), "n_in": n_in},
        n_classes=n_out,
        seed=42,
    )
    p = model.predict_proba(x.numpy())
    assert p.shape == (n_samples, n_out)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)


def test_eval_mode_deterministic() -> None:
    """Kiểm tra ở chế độ eval(), hai lần gọi forward liên tiếp cho kết quả bằng nhau tuyệt đối."""
    torch.manual_seed(42)
    n_samples = 20
    n_in = 21
    net = MLPNet(n_in=n_in, hidden=(128, 64, 32), n_out=3, dropout=(0.3, 0.3))
    net.eval()

    x = torch.randn(n_samples, n_in)
    with torch.no_grad():
        out1 = net(x)
        out2 = net(x)

    assert torch.equal(out1, out2), "Hai lần gọi forward ở chế độ eval() phải bằng nhau tuyệt đối."


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Kiểm tra chu trình save/load của MLPModel: xác suất sau khi nạp giống trước đó <= 1e-9."""
    torch.manual_seed(42)
    n_samples = 25
    n_in = 16
    n_classes = 3

    model = MLPModel(
        params={"hidden": (32, 16), "dropout": (0.1, 0.0), "n_in": n_in},
        n_classes=n_classes,
        seed=42,
    )

    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_in).astype(np.float32)

    p_before = model.predict_proba(X)

    save_dir = tmp_path / "mlp_checkpoint"
    model.save(save_dir)

    assert (save_dir / "model.pt").is_file(), "File model.pt không tồn tại sau khi save."

    loaded_model = MLPModel.load(save_dir)
    p_after = loaded_model.predict_proba(X)

    max_diff = float(np.max(np.abs(p_before - p_after)))
    assert max_diff <= 1e-9, f"Xác suất sau khi load lệch {max_diff} > 1e-9."

    # Kiểm tra thêm export_numpy hoạt động chính xác
    exported = loaded_model.export_numpy()
    assert "weights" in exported and "biases" in exported and "layers" in exported
    assert len(exported["weights"]) == 3  # (16->32, 32->16, 16->3)


def test_overfits_tiny_batch() -> None:
    """Kiểm tra mạng có thể overfit trên batch nhỏ (64 mẫu, 300 bước, loss giảm > 50%)."""
    torch.manual_seed(42)
    n_samples = 64
    n_in = 21
    n_out = 3

    X = torch.randn(n_samples, n_in)
    y = torch.randint(0, n_out, (n_samples,))

    net = MLPNet(n_in=n_in, hidden=(64, 32), n_out=n_out, dropout=(0.0, 0.0))
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    initial_loss: float | None = None
    final_loss: float = 0.0

    for step in range(300):
        optimizer.zero_grad(set_to_none=True)
        logits = net(X)
        loss = criterion(logits, y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert initial_loss is not None
    assert final_loss < 0.5 * initial_loss, (
        f"Loss không giảm quá 50%: ban đầu {initial_loss:.4f}, kết thúc {final_loss:.4f}"
    )
