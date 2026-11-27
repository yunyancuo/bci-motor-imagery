"""核心指标计算、分层交叉验证划分与防泄漏标准化单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    accuracy_score_np,
    apply_zscore,
    cohen_kappa_score_np,
    confusion_matrix_np,
    stratified_folds,
    zscore_stats,
)
from src.run_experiment import stratified_folds as run_exp_stratified_folds


def test_accuracy_score_basic_and_errors() -> None:
    """验证多分类准确率计算与异常输入校验。"""
    y_true = np.array([0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 0, 2])
    assert accuracy_score_np(y_true, y_pred) == pytest.approx(4 / 6)

    with pytest.raises(ValueError, match="不能为空"):
        accuracy_score_np([], [])
    with pytest.raises(ValueError, match="长度不一致"):
        accuracy_score_np([0, 1], [0])
    with pytest.raises(ValueError, match="一维数组"):
        accuracy_score_np(np.zeros((2, 2)), np.zeros((2, 2)))


def test_cohen_kappa_score_analytical_and_degenerate() -> None:
    """验证 Cohen's kappa 的解析正确性以及单类别退化零分母安全保护。"""
    y_true = np.array([0, 1, 2, 3] * 10)
    assert cohen_kappa_score_np(y_true, y_true) == pytest.approx(1.0)

    # 四类完全均衡预测全错且边际分布均匀时，p_o = 0, p_e = 0.25 => kappa = -1/3
    y_shifted = (y_true + 1) % 4
    assert cohen_kappa_score_np(y_true, y_shifted) == pytest.approx(-1.0 / 3.0)

    # 全部样本同属一类且全部预测同类，p_e = 1.0，必须安全返回 0.0 而非 NaN
    y_const = np.zeros(12, dtype=np.int64)
    kappa = cohen_kappa_score_np(y_const, y_const)
    assert np.isfinite(kappa) and kappa == 0.0

    with pytest.raises(ValueError, match="不能为空"):
        cohen_kappa_score_np([], [])


def test_cohen_kappa_matches_sklearn_if_available() -> None:
    """若环境安装了 scikit-learn，校验纯 NumPy 实现与其数值严格一致。"""
    sk_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 4, size=200)
    y_pred = rng.integers(0, 4, size=200)

    ours = cohen_kappa_score_np(y_true, y_pred)
    ref = float(sk_metrics.cohen_kappa_score(y_true, y_pred))
    assert ours == pytest.approx(ref, abs=1e-12)


def test_confusion_matrix_handles_missing_class_without_nan() -> None:
    """验证行归一化混淆矩阵在某类别样本数为 0 时不会产生 NaN。"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 2])
    cm_raw = confusion_matrix_np(y_true, y_pred, n_classes=4, normalize=None)
    assert cm_raw.shape == (4, 4)
    assert cm_raw[0, 0] == 1 and cm_raw[0, 1] == 1
    assert cm_raw[2].sum() == 0 and cm_raw[3].sum() == 0

    cm_norm = confusion_matrix_np(y_true, y_pred, n_classes=4, normalize="true")
    assert np.all(np.isfinite(cm_norm))
    assert cm_norm[0, 0] == pytest.approx(0.5)
    assert cm_norm[0, 1] == pytest.approx(0.5)
    assert np.all(cm_norm[2] == 0.0)
    assert np.all(cm_norm[3] == 0.0)


def test_stratified_folds_balance_reproducibility_and_disjointness() -> None:
    """验证分层 k 折划分无泄漏、无重叠、类别均衡且与 run_experiment 重导出一致。"""
    y = np.repeat(np.arange(4), 25)  # 每类 25 个，共 100 个样本
    folds_a = stratified_folds(y, k=5, seed=42)
    folds_b = run_exp_stratified_folds(y, k=5, seed=42)

    assert len(folds_a) == 5
    for fa, fb in zip(folds_a, folds_b, strict=True):
        assert np.array_equal(fa, fb)
        assert len(fa) == 20
        counts = np.bincount(y[fa], minlength=4)
        assert np.all(counts == 5)

    all_indices = np.concatenate(folds_a)
    assert len(all_indices) == len(y)
    assert np.array_equal(np.sort(all_indices), np.arange(len(y)))

    with pytest.raises(ValueError, match="k 必须 >= 2"):
        stratified_folds(y, k=1)
    with pytest.raises(ValueError, match="小于折数"):
        stratified_folds(np.array([0, 1, 2]), k=5)


def test_zscore_stats_no_data_leakage_and_constant_channel() -> None:
    """验证通道标准化仅依赖训练集统计量（无测试集泄漏）且支持常数通道防除零。"""
    rng = np.random.default_rng(123)
    x_tr = rng.normal(loc=5.0, scale=2.0, size=(16, 4, 100)).astype(np.float32)
    x_tr[:, 3, :] = 7.0  # 常数通道，标准差为 0

    mu, sd = zscore_stats(x_tr)
    assert mu.shape == (1, 4, 1)
    assert sd.shape == (1, 4, 1)
    assert np.all(sd > 0)

    x_tr_norm = apply_zscore(x_tr, mu, sd)
    assert np.all(np.isfinite(x_tr_norm))
    assert np.abs(x_tr_norm[:, :3, :].mean(axis=(0, 2))).max() < 1e-5
    assert np.abs(x_tr_norm[:, :3, :].std(axis=(0, 2)) - 1.0).max() < 1e-4

    # 使用同一训练集统计量变换两组差异巨大的测试集，验证统计量与变换公式完全独立
    x_te_1 = np.zeros((4, 4, 100), dtype=np.float32)
    x_te_2 = np.concatenate([x_te_1[:1], np.full((3, 4, 100), 1e5, dtype=np.float32)], axis=0)
    out_1 = apply_zscore(x_te_1, mu, sd)[0]
    out_2 = apply_zscore(x_te_2, mu, sd)[0]
    assert np.allclose(out_1, out_2, atol=1e-6)
