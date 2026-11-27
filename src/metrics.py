"""核心指标、数据划分与归一化纯 NumPy 工具模块。

本模块仅依赖 Python 标准库与 NumPy，供所有训练、评估与集成脚本复用，
既避免在轻量化任务中触发重型深度学习库导入，也保证数值边界计算安全可靠。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

CLASS_NAMES: list[str] = ["左手", "右手", "双脚", "舌头"]


def accuracy_score_np(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
) -> float:
    """计算多分类准确率。

    参数:
        y_true: 真实类别标签一维数组。
        y_pred: 预测类别标签一维数组。

    返回:
        准确率浮点数（范围 ``[0.0, 1.0]``）。

    异常:
        ValueError: 输入为空或长度不一致时抛出。
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError(f"标签必须为一维数组，当前形状: y_true={yt.shape}, y_pred={yp.shape}")
    if yt.size == 0:
        raise ValueError("标签数组不能为空")
    if yt.shape != yp.shape:
        raise ValueError(f"真实标签与预测标签长度不一致: {yt.shape[0]} vs {yp.shape[0]}")
    return float(np.mean(yt == yp))


def confusion_matrix_np(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    n_classes: int | None = None,
    labels: Sequence[int] | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    """计算多分类混淆矩阵，支持安全的行归一化（避免零样本类别产生 NaN）。

    参数:
        y_true: 真实类别标签一维数组。
        y_pred: 预测类别标签一维数组。
        n_classes: 类别总数；若未指定且 ``labels`` 为空，则自动推断所有出现类别。
        labels: 显式指定的有序类别标签列表，优先级高于 ``n_classes``。
        normalize: 归一化方式，支持 ``None``（计数）或 ``"true"``（按真实类行归一化）。

    返回:
        形状为 ``(C, C)`` 的混淆矩阵数组（未归一化时为 ``int64``，归一化后为 ``float64``）。
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1 or yt.shape != yp.shape:
        raise ValueError(f"y_true 与 y_pred 必须为等长一维数组，当前: {yt.shape} vs {yp.shape}")
    if normalize not in (None, "true"):
        raise ValueError(f"不支持的 normalize 模式: {normalize!r}，仅支持 None 或 'true'")

    if labels is not None:
        label_list = list(labels)
    elif n_classes is not None:
        if n_classes <= 0:
            raise ValueError(f"n_classes 必须为正整数，当前为 {n_classes}")
        label_list = list(range(n_classes))
    else:
        label_list = sorted(set(yt.tolist()) | set(yp.tolist()))

    label_to_idx = {lab: idx for idx, lab in enumerate(label_list)}
    size = len(label_list)
    cm = np.zeros((size, size), dtype=np.int64)
    for t_val, p_val in zip(yt.tolist(), yp.tolist(), strict=True):
        if t_val in label_to_idx and p_val in label_to_idx:
            cm[label_to_idx[t_val], label_to_idx[p_val]] += 1

    if normalize == "true":
        cm_float = cm.astype(np.float64)
        row_sums = cm_float.sum(axis=1, keepdims=True)
        return np.divide(
            cm_float,
            row_sums,
            out=np.zeros_like(cm_float),
            where=row_sums > 0,
        )
    return cm


def cohen_kappa_score_np(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    labels: Sequence[int] | None = None,
) -> float:
    """计算 Cohen's kappa 一致性系数（与 scikit-learn 数值严格一致）。

    参数:
        y_true: 真实类别标签一维数组。
        y_pred: 预测类别标签一维数组。
        labels: 可选的显式类别标签列表。

    返回:
        Cohen's kappa 浮点值；当完全退化为单类且 ``p_e == 1`` 时返回 ``0.0`` 以防除零。
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.size == 0:
        raise ValueError("计算 kappa 时标签数组不能为空")
    cm = confusion_matrix_np(yt, yp, labels=labels, normalize=None).astype(np.float64)
    total = cm.sum()
    if total == 0:
        return 0.0
    p_o = np.trace(cm) / total
    row_marginal = cm.sum(axis=1)
    col_marginal = cm.sum(axis=0)
    p_e = float(np.dot(row_marginal, col_marginal)) / (total * total)
    denom = 1.0 - p_e
    if abs(denom) < 1e-12:
        return 0.0
    return float((p_o - p_e) / denom)


def stratified_folds(
    y: np.ndarray | Sequence[int],
    k: int = 5,
    seed: int = 42,
) -> list[np.ndarray]:
    """生成分层 k 折索引列表，使每折各类样本数量尽可能均衡。

    注意：保持 ``np.random.default_rng(seed)`` 的类别迭代与洗牌顺序不变，
    以确保与历史实验复现划分逐索引完全一致。

    参数:
        y: 样本标签一维数组。
        k: 折数，必须满足 ``2 <= k <= len(y)``。
        seed: 随机种子，默认 ``42``。

    返回:
        长度为 ``k`` 的列表，每个元素为升序排列的测试样本索引 ``np.ndarray``。
    """
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y 必须为一维数组，当前维度: {y_arr.ndim}")
    if k < 2:
        raise ValueError(f"交叉验证折数 k 必须 >= 2，当前为 {k}")
    if y_arr.size < k:
        raise ValueError(f"样本总数 ({y_arr.size}) 小于折数 k ({k})")

    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in np.unique(y_arr):
        idx = np.where(y_arr == cls)[0]
        rng.shuffle(idx)
        for i, sample_idx in enumerate(idx):
            folds[i % k].append(int(sample_idx))
    return [np.array(sorted(fold), dtype=np.int64) for fold in folds]


def zscore_stats(
    X: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """在训练集上按通道计算 Z-score 统计量（均值与标准差）。

    仅在训练集上调用本函数并将返回的 ``(mu, sd)`` 应用于测试集，
    可严格避免跨样本/跨折的数据泄漏。

    参数:
        X: 三维脑电张量，形状为 ``(n_trials, n_channels, n_times)``。
        eps: 防止除零的平滑常数，默认 ``1e-6``。

    返回:
        ``(mu, sd)`` 元组，形状均为 ``(1, n_channels, 1)``。
    """
    x_arr = np.asarray(X, dtype=np.float64)
    if x_arr.ndim != 3:
        raise ValueError(f"输入 EEG 张量必须为 3D (N, C, T)，当前形状: {x_arr.shape}")
    if x_arr.shape[0] == 0 or x_arr.shape[2] == 0:
        raise ValueError(f"输入 EEG 张量不能包含空维度，当前形状: {x_arr.shape}")
    mu = x_arr.mean(axis=(0, 2), keepdims=True)
    sd = x_arr.std(axis=(0, 2), keepdims=True) + eps
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_zscore(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """使用预先拟合的均值与标准差对 EEG 数据执行标准化。

    参数:
        X: 待标准化三维 EEG 张量 ``(N, C, T)``。
        mu: 训练集通道均值 ``(1, C, 1)``。
        sd: 训练集通道标准差 ``(1, C, 1)``。

    返回:
        标准化后的 ``float32`` 张量。
    """
    x_arr = np.asarray(X, dtype=np.float32)
    return ((x_arr - mu) / sd).astype(np.float32)
