"""Đánh giá mô hình bằng kiểm định chéo K-fold phân tầng (Stratified K-Fold).

Chỉ sử dụng tập huấn luyện (train), tuyệt đối không đụng đến tập validation hoặc test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from diabetes_ai.evaluation.metrics import roc_auc_ovr
from diabetes_ai.models.registry import build_model

if TYPE_CHECKING:
    from diabetes_ai.features.build_features import ExperimentData


def stratified_kfold_auc(
    model_key: str,
    params: dict[str, Any] | None,
    data: ExperimentData | Any,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Ước lượng độ ổn định của mô hình qua Stratified K-Fold AUC trên tập train.

    Chỉ nạp và chia nhỏ trên data.X_train và data.y_train; không truy cập validation hay test.

    Args:
        model_key: Tên nhận diện mô hình ('majority', 'logreg', 'xgboost', ...).
        params: Siêu tham số khởi tạo mô hình.
        data: Đối tượng chứa dữ liệu thí nghiệm (cần có X_train và y_train).
        n_splits: Số phân đoạn K-fold (mặc định 5).
        seed: Seed sinh số ngẫu nhiên đảm bảo tính tái lập.

    Returns:
        dict[str, Any]: Gồm 'mean' (AUC trung bình), 'std' (độ lệch chuẩn), và 'folds' (danh sách AUC).
    """
    X_train = np.asarray(data.X_train)
    y_train = np.asarray(data.y_train)

    if params is None:
        model_params: dict[str, Any] = {}
    else:
        model_params = dict(params)

    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_auc: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        model = build_model(
            model_key=model_key,
            params=model_params,
            n_classes=n_classes,
            seed=seed + fold_idx,
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)
        auc_val = float(roc_auc_ovr(y_va, proba))
        folds_auc.append(auc_val)

    mean_auc = float(np.mean(folds_auc))
    std_auc = float(np.std(folds_auc))

    return {
        "mean": mean_auc,
        "std": std_auc,
        "folds": folds_auc,
    }
