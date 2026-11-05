"""Biểu đồ đánh giá mô hình: ROC, PR, ma trận nhầm lẫn, hiệu chuẩn, đường học, công bằng.

Backend Agg được kích hoạt đầu file; độ phân giải chuẩn dpi=110; nhãn tiếng Việt.
Mọi hàm đều tự tạo thư mục cha nếu chưa có và trả về đối tượng Path của file đã lưu.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def plot_roc_ovr(
    y: np.ndarray | list[int],
    proba: np.ndarray,
    class_names: Sequence[str],
    path: Path | str,
) -> Path:
    """Vẽ đường cong ROC One-vs-Rest cho từng lớp và lưu ra file ảnh PNG.

    Args:
        y: Nhãn thực tế (1D).
        proba: Xác suất dự đoán (n_samples, n_classes) hoặc (n_samples,).
        class_names: Tên các lớp tương ứng.
        path: Đường dẫn lưu file ảnh PNG.

    Returns:
        Path: Đường dẫn tới file ảnh đã lưu.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_arr = np.asarray(y)
    p_arr = np.asarray(proba)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    n_classes = len(class_names)

    if p_arr.ndim == 1 or p_arr.shape[1] == 1:
        p1 = p_arr.ravel()
        fpr, tpr, _ = roc_curve(y_arr, p1)
        roc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[-1]} (AUC = {roc_val:.3f})")
    else:
        for c in range(n_classes):
            y_c = (y_arr == c).astype(int)
            if np.sum(y_c) > 0 and np.sum(y_c) < len(y_c):
                fpr, tpr, _ = roc_curve(y_c, p_arr[:, c])
                roc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC = {roc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Ngẫu nhiên (AUC = 0.500)")
    ax.set_xlabel("Tỉ lệ dương tính giả (FPR)")
    ax.set_ylabel("Tỉ lệ dương tính thật (TPR / Recall)")
    ax.set_title("Đường cong ROC (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_pr_ovr(
    y: np.ndarray | list[int],
    proba: np.ndarray,
    class_names: Sequence[str],
    path: Path | str,
) -> Path:
    """Vẽ đường cong Precision-Recall One-vs-Rest cho từng lớp và lưu PNG.

    Args:
        y: Nhãn thực tế (1D).
        proba: Xác suất dự đoán (n_samples, n_classes) hoặc (n_samples,).
        class_names: Tên các lớp tương ứng.
        path: Đường dẫn lưu file ảnh PNG.

    Returns:
        Path: Đường dẫn tới file ảnh đã lưu.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_arr = np.asarray(y)
    p_arr = np.asarray(proba)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    n_classes = len(class_names)

    if p_arr.ndim == 1 or p_arr.shape[1] == 1:
        p1 = p_arr.ravel()
        precision, recall, _ = precision_recall_curve(y_arr, p1)
        pr_val = auc(recall, precision)
        ax.plot(recall, precision, label=f"{class_names[-1]} (PR-AUC = {pr_val:.3f})")
    else:
        for c in range(n_classes):
            y_c = (y_arr == c).astype(int)
            if np.sum(y_c) > 0:
                precision, recall, _ = precision_recall_curve(y_c, p_arr[:, c])
                pr_val = auc(recall, precision)
                ax.plot(recall, precision, label=f"{class_names[c]} (PR-AUC = {pr_val:.3f})")

    ax.set_xlabel("Độ nhạy (Recall)")
    ax.set_ylabel("Độ chính xác (Precision)")
    ax.set_title("Đường cong Precision-Recall (One-vs-Rest)")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_confusion(
    cm: np.ndarray,
    class_names: Sequence[str],
    path: Path | str,
    normalize: str | None = "true",
) -> Path:
    """Vẽ ma trận nhầm lẫn có hiển thị nhãn và giá trị chuẩn hóa.

    Args:
        cm: Ma trận nhầm lẫn 2D.
        class_names: Danh sách tên lớp.
        path: Đường dẫn lưu file ảnh.
        normalize: Kiểu chuẩn hóa ("true", "pred", "all" hoặc None). Mặc định "true".

    Returns:
        Path: Đường dẫn tới file ảnh đã lưu.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.asarray(cm, dtype=float)
    if normalize == "true":
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)
    elif normalize == "pred":
        col_sums = cm_arr.sum(axis=0, keepdims=True)
        cm_norm = np.divide(cm_arr, col_sums, out=np.zeros_like(cm_arr), where=col_sums != 0)
    elif normalize == "all":
        total = cm_arr.sum()
        cm_norm = cm_arr / total if total > 0 else cm_arr
    else:
        cm_norm = cm_arr

    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    im = ax.imshow(
        cm_norm,
        interpolation="nearest",
        cmap=plt.cm.Blues,
        vmin=0,
        vmax=1.0 if normalize else None,
    )
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = (cm_norm.max() + cm_norm.min()) / 2.0 if cm_norm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_norm = cm_norm[i, j]
            val_raw = int(cm[i, j])
            text = f"{val_raw}\n({val_norm:.1%})" if normalize else f"{val_raw}"
            color = "white" if val_norm > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_title("Ma trận nhầm lẫn")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_reliability(
    y: np.ndarray | list[int],
    proba: np.ndarray,
    class_names: Sequence[str],
    path: Path | str,
    proba_before: np.ndarray | None = None,
    n_bins: int = 10,
) -> Path:
    """Vẽ biểu đồ độ tin cậy (reliability diagram / calibration curve).

    Hiển thị đường chéo tham chiếu lý tưởng, cùng đường trước và sau hiệu chuẩn.

    Args:
        y: Nhãn thực tế (1D).
        proba: Xác suất dự đoán sau hiệu chuẩn (n_samples, n_classes) hoặc (n_samples,).
        class_names: Tên các lớp tương ứng.
        path: Đường dẫn lưu file ảnh.
        proba_before: Xác suất dự đoán thô trước hiệu chuẩn (tùy chọn).
        n_bins: Số lượng bin chia xác suất (mặc định 10).

    Returns:
        Path: Đường dẫn tới file ảnh đã lưu.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_arr = np.asarray(y)
    p_arr = np.asarray(proba)
    pb_arr = np.asarray(proba_before) if proba_before is not None else None

    if p_arr.ndim == 1 or p_arr.shape[1] == 1:
        n_plots = 1
        classes_to_plot = [(1, class_names[-1])]
    else:
        n_plots = min(len(class_names), p_arr.shape[1])
        classes_to_plot = [(c, class_names[c]) for c in range(n_plots)]

    fig, axes = plt.subplots(
        1, n_plots, figsize=(4.5 * n_plots, 4.0), dpi=110, sharey=True, squeeze=False
    )
    axes_flat = axes.ravel()

    for idx, (c, name) in enumerate(classes_to_plot):
        ax = axes_flat[idx]
        y_c = (y_arr == c).astype(int) if p_arr.ndim > 1 and p_arr.shape[1] > 1 else y_arr
        p_c = p_arr[:, c] if p_arr.ndim > 1 and p_arr.shape[1] > 1 else p_arr.ravel()

        ax.plot([0, 1], [0, 1], "k--", label="Lý tưởng")

        if pb_arr is not None:
            pb_c = pb_arr[:, c] if pb_arr.ndim > 1 and pb_arr.shape[1] > 1 else pb_arr.ravel()
            try:
                prob_true_b, prob_pred_b = calibration_curve(y_c, pb_c, n_bins=n_bins, strategy="uniform")
                ax.plot(prob_pred_b, prob_true_b, "s--", color="#d62728", label="Trước hiệu chuẩn")
            except Exception:
                pass

        try:
            prob_true, prob_pred = calibration_curve(y_c, p_c, n_bins=n_bins, strategy="uniform")
            label_after = "Sau hiệu chuẩn" if pb_arr is not None else "Đã hiệu chuẩn"
            ax.plot(prob_pred, prob_true, "o-", color="#1f77b4", label=label_after)
        except Exception:
            pass

        ax.set_xlabel("Xác suất dự đoán")
        if idx == 0:
            ax.set_ylabel("Tần suất thực tế")
        ax.set_title(f"Lớp: {name}")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_learning_curve(
    history: list[dict[str, Any]],
    path: Path | str,
    best_epoch: int | None = None,
) -> Path:
    """Vẽ đường học tập (learning curve) gồm Loss và Validation AUC theo epoch.

    Args:
        history: Danh sách từ điển chứa số đo từng epoch (loss, val_auc,...).
        path: Đường dẫn lưu file ảnh.
        best_epoch: Epoch được chọn (nếu None sẽ tự xác định epoch có val_auc cao nhất).

    Returns:
        Path: Đường dẫn tới file ảnh đã lưu.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    train_loss = [h.get("train_loss", np.nan) for h in history]
    val_loss = [h.get("val_loss", np.nan) for h in history]
    val_auc = [h.get("val_auc_ovr", h.get("val_auc", np.nan)) for h in history]

    if best_epoch is None:
        valid_aucs = [(e, a) for e, a in zip(epochs, val_auc) if not np.isnan(a)]
        if valid_aucs:
            best_epoch = max(valid_aucs, key=lambda x: x[1])[0]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5), dpi=110)

    color_train = "#1f77b4"
    color_val = "#ff7f0e"
    ax1.plot(epochs, train_loss, label="Train Loss", color=color_train, linestyle="-")
    ax1.plot(epochs, val_loss, label="Val Loss", color=color_val, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Hàm mất mát (Cross-Entropy Loss)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_auc = "#2ca02c"
    ax2.plot(epochs, val_auc, label="Val AUC (OvR)", color=color_auc, linewidth=2)
    ax2.set_ylabel("Validation AUC", color=color_auc)
    ax2.tick_params(axis="y", labelcolor=color_auc)

    if best_epoch is not None and best_epoch in epochs:
        ax1.axvline(
            x=best_epoch,
            color="#d62728",
            linestyle=":",
            linewidth=2,
            label=f"Epoch chọn ({best_epoch})",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

    plt.title("Đường học tập theo Epoch (MLP)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_fairness_bars(
    rows_by_group: dict[str, Any] | list[dict[str, Any]],
    metric: str,
    path: Path | str,
) -> Path:
    """Vẽ biểu đồ cột so sánh chỉ số giữa các nhóm nhân khẩu học kèm khoảng tin cậy Wilson.

    Args:
        rows_by_group: Từ điển các nhóm (ví dụ {"Age": rows, "Sex": rows}) hoặc danh sách row.
        metric: Tên chỉ số cần vẽ (ví dụ "recall", "precision", "alert_rate").
        path: Đường dẫn lưu file ảnh.

    Returns:
        Path: Đường dẫn tới file ảnh đã lưu.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    categories: dict[str, list[dict[str, Any]]] = {}
    if isinstance(rows_by_group, list):
        categories["Nhóm"] = rows_by_group
    elif isinstance(rows_by_group, dict):
        for k, v in rows_by_group.items():
            if k in ("min_pos", "disparity", "max_gap_pp", "by_class"):
                continue
            if isinstance(v, dict) and "rows" in v:
                categories[k] = v["rows"]
            elif isinstance(v, list):
                categories[k] = v

    if not categories:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=110)
        ax.text(0.5, 0.5, "Không có dữ liệu", ha="center", va="center")
        plt.savefig(out_path, dpi=110)
        plt.close(fig)
        return out_path

    n_cats = len(categories)
    fig, axes = plt.subplots(
        1, n_cats, figsize=(4.5 * n_cats, 4.0), dpi=110, sharey=True, squeeze=False
    )
    axes_flat = axes.ravel()

    metric_names_vi = {
        "recall": "Độ nhạy (Recall)",
        "precision": "Độ chuẩn xác (Precision)",
        "prevalence": "Tỉ lệ hiện mắc (Prevalence)",
        "alert_rate": "Tỉ lệ cảnh báo (Alert rate)",
    }
    metric_label = metric_names_vi.get(metric.lower(), metric.capitalize())

    for idx, (cat_name, rows) in enumerate(categories.items()):
        ax = axes_flat[idx]
        groups: list[str] = []
        points: list[float] = []
        yerr_lower: list[float] = []
        yerr_upper: list[float] = []
        bar_colors: list[str] = []

        for r in rows:
            g = str(r.get("group", ""))
            groups.append(g)
            m_val = r.get(metric)
            if isinstance(m_val, dict):
                pt = float(m_val.get("point", 0.0))
                low = float(m_val.get("low", pt))
                high = float(m_val.get("high", pt))
            elif isinstance(m_val, (int, float)):
                pt = float(m_val)
                low = pt
                high = pt
            else:
                pt = 0.0
                low = 0.0
                high = 0.0

            points.append(pt)
            yerr_lower.append(max(0.0, pt - low))
            yerr_upper.append(max(0.0, high - pt))

            if r.get("small_sample", False):
                bar_colors.append("#aec7e8")
            else:
                bar_colors.append("#1f77b4")

        x = np.arange(len(groups))
        yerr = [yerr_lower, yerr_upper]
        ax.bar(
            x,
            points,
            yerr=yerr,
            capsize=4,
            color=bar_colors,
            edgecolor="black",
            alpha=0.85,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"Nhóm: {cat_name}")
        if idx == 0:
            ax.set_ylabel(f"{metric_label} kèm khoảng Wilson 95%")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(f"Công bằng giữa các nhóm — {metric_label}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
