"""统一中文字体设置：优先加载项目自带 SimHei，按平台顺序优雅回退系统字体。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager

DEFAULT_FONT_PATH = Path(__file__).resolve().parent.parent / "fonts" / "simhei.ttf"
FALLBACK_FONTS: list[str] = [
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]


def configure_chinese_font(font_path: Path | str | None = None) -> list[str]:
    """配置 Matplotlib 中文字体与负号显示，避免中文标题乱码或负号方框。

    参数:
        font_path: 可选的本地字体文件路径；若未提供则自动探测项目内置 ``fonts/simhei.ttf``。

    返回:
        实际写入 ``matplotlib.rcParams["font.sans-serif"]`` 的字体优先级列表。
    """
    target = Path(font_path) if font_path is not None else DEFAULT_FONT_PATH
    loaded_local = False
    if target.exists() and target.is_file():
        try:
            font_manager.fontManager.addfont(str(target))
            loaded_local = True
        except (OSError, RuntimeError):
            # 字体文件损坏或不可读时降级使用系统回退字体
            loaded_local = False

    if loaded_local:
        fonts = list(FALLBACK_FONTS)
    else:
        fonts = [f for f in FALLBACK_FONTS if f != "SimHei"] + ["SimHei"]

    matplotlib.rcParams["font.sans-serif"] = fonts
    matplotlib.rcParams["axes.unicode_minus"] = False
    return fonts


# 保持向后兼容：导入本模块即自动完成中文字体注册
configure_chinese_font()
