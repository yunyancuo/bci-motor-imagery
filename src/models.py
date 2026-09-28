"""模型定义：ShallowConvNet（改进）与 EEGNet（对照）。

ShallowConvNet（Schirrmeister et al., 2017）：时间卷积 + 空间深度卷积 + 平方非线性
+ 平均池化 + 对数压缩，把 CSP"空间滤波 + 带通 + log 方差特征"的整套流程搬进网络
端到端学习，在运动想象小数据上通常优于 EEGNet。
"""
import torch
import torch.nn as nn


class _SquareLog(nn.Module):
    def forward(self, x):
        return torch.log(x.clamp(min=1e-6))


class ShallowConvNet(nn.Module):
    """标准顺序：时间卷积 → 空间深度卷积 → 平方 → 平均池化 → 对数 → 全连接。"""

    def __init__(self, n_ch=22, n_classes=4, n_times=500, n_filters=40, drop=0.4):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters, (1, 25), padding=(0, 12), bias=True)
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_ch, 1), groups=n_filters, bias=False)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop = nn.Dropout(drop)
        feat = n_filters * (((n_times - 75) // 15) + 1)
        self.fc = nn.Linear(feat, n_classes)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.spatial(self.temporal(x))
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(x.clamp(min=1e-6))
        return self.fc(self.drop(x.flatten(1)))


class EEGNet(nn.Module):
    def __init__(self, n_ch=22, n_classes=4, fs=250, d1=8, d2=16, kern_len=125, dropout=0.5):
        super().__init__()
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
        feat = 16 * ((n_times_hint(fs)) // 32)
        self.fc = nn.Linear(feat, n_classes)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(x.flatten(1))


def n_times_hint(fs, n_times=500):
    return n_times
