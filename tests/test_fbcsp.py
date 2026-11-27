"""FBCSP 滤波器组稳定性、特征维度与零数据泄漏专项测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.fbcsp import BANDS, FBCSP, bandpass_filter, run_fbcsp


def test_bandpass_filter_frequency_selectivity_and_dtype() -> None:
    """验证带通滤波器能精准保留目标子带信号并强衰减带外频率分量。"""
    fs = 250.0
    n_times = 1000
    t = np.arange(n_times) / fs
    sig_10hz = np.sin(2 * np.pi * 10.0 * t)
    X = np.tile(sig_10hz, (2, 3, 1)).astype(np.float32)

    # 通过 8–12 Hz 子带应保留绝大部分能量；通过 24–28 Hz 子带应被几乎完全抑制
    in_band = bandpass_filter(X, 8, 12, fs=fs)
    out_band = bandpass_filter(X, 24, 28, fs=fs)

    assert in_band.shape == X.shape
    assert in_band.dtype == np.float32
    assert np.all(np.isfinite(in_band))

    # 取稳态中段计算功率比
    mid = slice(200, 800)
    in_power = float(np.mean(in_band[:, :, mid] ** 2))
    out_power = float(np.mean(out_band[:, :, mid] ** 2))
    orig_power = float(np.mean(X[:, :, mid] ** 2))

    assert in_power / orig_power > 0.90
    assert out_power / orig_power < 1e-3


def test_bandpass_filter_validation_errors() -> None:
    """验证非法频率、采样率、维度或过短时间窗时给出清晰异常。"""
    X = np.zeros((2, 4, 200), dtype=np.float32)
    with pytest.raises(ValueError, match="3D"):
        bandpass_filter(np.zeros((4, 200)), 8, 12)
    with pytest.raises(ValueError, match="0 < lo < hi < fs/2"):
        bandpass_filter(X, 12, 8, fs=250)
    with pytest.raises(ValueError, match="0 < lo < hi < fs/2"):
        bandpass_filter(X, 8, 130, fs=250)
    with pytest.raises(ValueError, match="时间长度过短"):
        bandpass_filter(np.zeros((2, 4, 16)), 8, 12, fs=250)


def test_fbcsp_feature_dimensions_and_unfitted_guard(
    synthetic_eeg: tuple[np.ndarray, np.ndarray],
) -> None:
    """验证 FBCSP 默认 6×4=24 维特征输出、自定义频段维度以及未拟合保护。"""
    X, y = synthetic_eeg
    fbcsp = FBCSP(n_components=4)

    with pytest.raises(RuntimeError, match="尚未拟合"):
        fbcsp.transform(X)

    feats = fbcsp.fit_transform(X, y)
    assert feats.shape == (X.shape[0], len(BANDS) * 4)
    assert np.all(np.isfinite(feats))

    # 自定义 3 个子带 × 2 个分量 => 6 维特征
    custom = FBCSP(n_components=2, bands=[(8, 12), (16, 20), (24, 28)])
    custom_feats = custom.fit_transform(X, y)
    assert custom_feats.shape == (X.shape[0], 6)
    assert np.all(np.isfinite(custom_feats))


def test_fbcsp_strict_no_test_leakage(
    synthetic_eeg: tuple[np.ndarray, np.ndarray],
) -> None:
    """验证 FBCSP.transform 对单条测试样本的输出不受同批其他测试样本污染（零泄漏）。"""
    X, y = synthetic_eeg
    X_tr, y_tr = X[:36], y[:36]
    X_probe = X[36:38]

    model = FBCSP(n_components=4).fit(X_tr, y_tr)
    feat_clean = model.transform(X_probe)

    # 往同批次拼入极端异常值样本，验证探针样本提取的特征逐元素不变
    outliers = np.full((6, X.shape[1], X.shape[2]), 1e5, dtype=np.float32)
    batch_dirty = np.concatenate([X_probe, outliers], axis=0)
    feat_dirty = model.transform(batch_dirty)[: len(X_probe)]

    assert np.allclose(feat_clean, feat_dirty, atol=1e-6)


def test_run_fbcsp_pipeline_on_synthetic_eeg(
    synthetic_eeg: tuple[np.ndarray, np.ndarray],
) -> None:
    """若环境安装了 scikit-learn，验证端到端 FBCSP + LDA 在可分合成 EEG 上高准确分类。"""
    pytest.importorskip("sklearn")
    X, y = synthetic_eeg

    # 每类取前 9 个作训练，后 3 个作测试（严格分层且无泄漏）
    tr_idx, te_idx = [], []
    for cls in range(4):
        cls_idx = np.where(y == cls)[0]
        tr_idx.append(cls_idx[:9])
        te_idx.append(cls_idx[9:])
    tr_idx_arr = np.concatenate(tr_idx)
    te_idx_arr = np.concatenate(te_idx)

    acc, kappa, pred, prob = run_fbcsp(
        X[tr_idx_arr], y[tr_idx_arr], X[te_idx_arr], y[te_idx_arr], n_components=4
    )
    assert pred.shape == (len(te_idx_arr),)
    assert prob.shape == (len(te_idx_arr), 4)
    assert np.allclose(prob.sum(axis=1), 1.0, atol=1e-5)
    assert acc >= 0.75
    assert kappa >= 0.60
