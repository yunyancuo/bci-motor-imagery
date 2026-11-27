"""深度学习模型定义：ShallowConvNet（改进对照）与 EEGNet（经典对照）。

ShallowConvNet（Schirrmeister et al., 2017）采用“时间卷积 + 空间深度卷积 + 平方非线性
+ 平均池化 + 对数压缩”结构，将经典 CSP“带通滤波 + 空间滤波 + log 方差特征”的信号处理
流程端到端映射到神经网络中，在运动想象小样本场景下具备良好的归纳偏置。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    """ShallowConvNet 网络结构：时间卷积 → 空间深度卷积 → 平方 → 平均池化 → 对数 → 全连接。"""

    def __init__(
        self,
        n_ch: int = 22,
        n_classes: int = 4,
        n_times: int = 500,
        n_filters: int = 40,
        drop: float = 0.4,
    ) -> None:
        """初始化 ShallowConvNet 模型参数。

        参数:
            n_ch: 输入脑电通道数，BCI IV 2a 为 22。
            n_classes: 分类类别数，四分类任务为 4。
            n_times: 输入时间采样点长度，必须满足 ``n_times >= 75``。
            n_filters: 时间/空间卷积滤波器数量，默认 40。
            drop: 全连接层前的 Dropout 失活比例，范围 ``[0.0, 1.0)``。
        """
        super().__init__()
        if n_ch < 1:
            raise ValueError(f"通道数 n_ch 必须 >= 1，当前为 {n_ch}")
        if n_classes < 2:
            raise ValueError(f"类别数 n_classes 必须 >= 2，当前为 {n_classes}")
        if n_filters < 1:
            raise ValueError(f"滤波器数 n_filters 必须 >= 1，当前为 {n_filters}")
        if n_times < 75:
            raise ValueError(f"时间点长度 n_times 必须 >= 75（池化核宽），当前为 {n_times}")
        if not 0.0 <= drop < 1.0:
            raise ValueError(f"drop 必须在 [0.0, 1.0) 范围内，当前为 {drop}")

        self.n_ch = n_ch
        self.n_classes = n_classes
        self.n_times = n_times
        self.n_filters = n_filters

        self.temporal = nn.Conv2d(1, n_filters, (1, 25), padding=(0, 12), bias=True)
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_ch, 1), groups=n_filters, bias=False)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop = nn.Dropout(drop)

        feat_dim = n_filters * (((n_times - 75) // 15) + 1)
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行前向推理。

        参数:
            x: 形状为 ``(batch, n_ch, n_times)`` 的输入脑电张量。

        返回:
            形状为 ``(batch, n_classes)`` 的未归一化分类 logits。
        """
        if x.ndim != 3:
            raise ValueError(f"ShallowConvNet 期望 3D 输入 (B, C, T)，当前形状: {tuple(x.shape)}")
        if x.shape[1] != self.n_ch:
            raise ValueError(f"输入通道数不匹配：期望 {self.n_ch}，实际 {x.shape[1]}")
        x = x[:, None, :, :]
        x = self.spatial(self.temporal(x))
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(x.clamp(min=1e-6))
        return self.fc(self.drop(x.flatten(1)))


class EEGNet(nn.Module):
    """EEGNet 紧凑型深度可分离卷积网络（Lawhern et al., 2018 风格实现）。"""

    def __init__(
        self,
        n_ch: int = 22,
        n_classes: int = 4,
        fs: int = 250,
        d1: int = 8,
        d2: int = 16,
        kern_len: int = 125,
        dropout: float = 0.5,
        n_times: int | None = None,
    ) -> None:
        """初始化 EEGNet 模型参数，并自动推导全连接层输入维度。

        参数:
            n_ch: 脑电通道数，默认 22。
            n_classes: 类别数，默认 4。
            fs: 采样率（Hz），默认 250，保留用于兼容历史签名。
            d1: 第一阶段时间滤波器数，默认 8。
            d2: 第二阶段卷积输出通道数，默认 16。
            kern_len: 第一阶段时间卷积核长度，默认 125（约 0.5 秒 @250Hz）。
            dropout: Dropout 失活比例，默认 0.5。
            n_times: 输入时间采样点长度；未提供时沿用 ``n_times_hint(fs)`` 默认 500。
        """
        super().__init__()
        if n_ch < 1 or n_classes < 2 or d1 < 1 or d2 < 1 or kern_len < 1:
            raise ValueError("网络通道数、类别数、滤波器数与卷积核长度均需满足正整数约束")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout 必须在 [0.0, 1.0) 范围内，当前为 {dropout}")

        resolved_times = n_times_hint(fs) if n_times is None else n_times
        if resolved_times < 32:
            raise ValueError(f"时间点长度 n_times 必须 >= 32，当前为 {resolved_times}")

        self.n_ch = n_ch
        self.n_classes = n_classes
        self.n_times = resolved_times

        self.block1 = nn.Sequential(
            nn.Conv2d(1, d1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(d1),
            nn.Conv2d(d1, d1 * 2, (n_ch, 1), groups=d1, bias=False),
            nn.BatchNorm2d(d1 * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(d1 * 2, d2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(d2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # 通过实际形状推算全连接层维度，修复硬编码 16 与偶数卷积核造成的维度偏移 Bug
        feat_dim = self._infer_feature_dim(n_ch, resolved_times)
        self.fc = nn.Linear(feat_dim, n_classes)

    def _infer_feature_dim(self, n_ch: int, n_times: int) -> int:
        """在不影响 BatchNorm 统计量的前提下精确推导展平特征维度。"""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_times)
            out = self.block2(self.block1(dummy))
        if was_training:
            self.train()
        return int(out.flatten(1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行前向推理。

        参数:
            x: 形状为 ``(batch, n_ch, n_times)`` 的输入脑电张量。

        返回:
            形状为 ``(batch, n_classes)`` 的未归一化分类 logits。
        """
        if x.ndim != 3:
            raise ValueError(f"EEGNet 期望 3D 输入 (B, C, T)，当前形状: {tuple(x.shape)}")
        if x.shape[1] != self.n_ch:
            raise ValueError(f"输入通道数不匹配：期望 {self.n_ch}，实际 {x.shape[1]}")
        x = x[:, None, :, :]
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(x.flatten(1))


def n_times_hint(fs: int, n_times: int = 500) -> int:
    """返回默认时间窗口采样点长度提示（保持向后兼容接口）。"""
    _ = fs
    return n_times
