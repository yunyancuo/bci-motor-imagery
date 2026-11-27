"""深度学习模型（ShallowConvNet / EEGNet）与训练流程单元测试。

若当前环境未安装 PyTorch，本文件会自动跳过并在安装了 CPU PyTorch 的 CI 环境中全量运行。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models import EEGNet, ShallowConvNet, n_times_hint  # noqa: E402
from src.run_experiment import run_shallow  # noqa: E402


def test_shallow_convnet_forward_backward_and_validation() -> None:
    """验证 ShallowConvNet 在默认与实验窗口配置下的前向、反传及参数校验。"""
    for n_ch, n_times, n_filters in [(22, 500, 40), (22, 750, 60), (8, 250, 16)]:
        model = ShallowConvNet(n_ch=n_ch, n_classes=4, n_times=n_times, n_filters=n_filters)
        x = torch.randn(4, n_ch, n_times)
        logits = model(x)
        assert logits.shape == (4, 4)
        assert torch.isfinite(logits).all()
        logits.sum().backward()

    with pytest.raises(ValueError, match="n_times 必须 >= 75"):
        ShallowConvNet(n_times=50)

    model = ShallowConvNet(n_ch=8, n_times=200)
    with pytest.raises(ValueError, match="输入通道数不匹配"):
        model(torch.randn(2, 6, 200))


def test_eegnet_handles_custom_d2_and_arbitrary_n_times() -> None:
    """回归测试：修复 EEGNet 硬编码 16 与偶数卷积核长度造成的维度不匹配 Bug。"""
    assert n_times_hint(250) == 500
    for d2, n_times in [(16, 500), (32, 316), (24, 512), (16, 750)]:
        model = EEGNet(n_ch=22, n_classes=4, d1=8, d2=d2, n_times=n_times)
        x = torch.randn(3, 22, n_times)
        logits = model(x)
        assert logits.shape == (3, 4)
        assert torch.isfinite(logits).all()


def test_run_shallow_reproducible_and_validates_window(
    synthetic_eeg: tuple[np.ndarray, np.ndarray],
) -> None:
    """验证 run_shallow 在 CPU 上端到端可复现运行，且能拦截滑窗越界错误。"""
    X, y = synthetic_eeg
    X_tr, y_tr = X[:32], y[:32]
    X_te, y_te = X[32:], y[32:]

    acc1, kap1, pred1, prob1 = run_shallow(
        X_tr,
        y_tr,
        X_te,
        y_te,
        epochs=2,
        seed=42,
        device=torch.device("cpu"),
        win=250,
        n_crops=2,
        n_filters=16,
        batch_size=16,
    )
    acc2, kap2, pred2, prob2 = run_shallow(
        X_tr,
        y_tr,
        X_te,
        y_te,
        epochs=2,
        seed=42,
        device=torch.device("cpu"),
        win=250,
        n_crops=2,
        n_filters=16,
        batch_size=16,
    )
    assert 0.0 <= acc1 <= 1.0
    assert np.isfinite(kap1)
    assert pred1.shape == y_te.shape
    assert np.allclose(prob1.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(pred1, pred2)
    assert np.allclose(prob1, prob2, atol=1e-6)
    assert acc1 == acc2 and kap1 == kap2

    with pytest.raises(ValueError, match="小于滑窗长度"):
        run_shallow(X_tr, y_tr, X_te, y_te, epochs=1, device=torch.device("cpu"), win=600)
