"""统一中文字体设置：优先用项目自带的 SimHei，回退服务器系统的文泉驿。"""
from pathlib import Path

import matplotlib
from matplotlib import font_manager

_FONT = Path(__file__).resolve().parent.parent / "fonts" / "simhei.ttf"
if _FONT.exists():
    font_manager.fontManager.addfont(str(_FONT))
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Zen Hei", "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
