"""Pytest 共享夹具：构造可复现的合成多通道运动想象 EEG 数据集。"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def synthetic_eeg() -> tuple[np.ndarray, np.ndarray]:
    """生成轻量级合成四类多通道 EEG 信号 ``(X, y)``。

    在不同类别下，向不同的空间电极子集与 μ/β 子带（10 Hz / 14 Hz / 22 Hz / 26 Hz）
    注入显著可分的振荡功率差异，用于离线验证滤波器组、CSP 特征与模型管线。
    """
    rng = np.random.default_rng(2026)
    fs = 250
    n_classes = 4
    n_per_class = 12
    n_ch = 8
    n_times = 500  # 2 秒 @250Hz

    t = np.arange(n_times, dtype=np.float64) / fs
    class_freqs = [10.0, 14.0, 22.0, 26.0]
    class_channels = [(0, 1), (2, 3), (4, 5), (6, 7)]

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for cls in range(n_classes):
        freq = class_freqs[cls]
        ch_pair = class_channels[cls]
        for _ in range(n_per_class):
            trial = rng.normal(0.0, 0.3, size=(n_ch, n_times))
            phase = rng.uniform(0, 2 * np.pi)
            wave = 2.5 * np.sin(2 * np.pi * freq * t + phase)
            for ch in ch_pair:
                trial[ch] += wave
            xs.append(trial)
            ys.append(cls)

    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y
