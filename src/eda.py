"""EDA：运动想象引起的 ERD/ERS 示意（C3/C4 左右手对比）与静息段功率谱密度图。

注意：BCI Competition IV 2a 的 GDF 通道名带 ``EEG-`` 前缀（如 ``EEG-C3``），
事件条件名映射为类别序号 0–3。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import src.cn_font  # noqa: F401  注册中文字体
    from src.metrics import CLASS_NAMES
    from src.prepare import parse_annotation_id_map
except ImportError:
    import cn_font  # noqa: F401  注册中文字体
    from metrics import CLASS_NAMES
    from prepare import parse_annotation_id_map

__all__ = ["CLASS_NAMES", "compute_erd_percentage", "main"]


def compute_erd_percentage(
    trials: np.ndarray,
    times: np.ndarray,
    baseline: tuple[float, float] = (-1.0, 0.0),
    eps: float = 1e-12,
) -> np.ndarray:
    """计算相对基线窗口的 ERD/ERS 能量变化百分比曲线。

    参数:
        trials: 单电极多 trial 信号，形状为 ``(n_trials, n_times)``。
        times: 对应的时间轴（秒），形状为 ``(n_times,)``。
        baseline: 基线窗口 ``(start_sec, end_sec)``，默认 ``(-1.0, 0.0)``。
        eps: 基线能量防除零下界阈值，默认 ``1e-12``。

    返回:
        形状为 ``(n_times,)`` 的 ERD/ERS 百分比变化曲线。
    """
    trials_arr = np.asarray(trials, dtype=np.float64)
    times_arr = np.asarray(times, dtype=np.float64)
    if trials_arr.ndim != 2 or trials_arr.shape[0] == 0:
        raise ValueError(f"trials 必须为非空 2D 数组 (n_trials, n_times)，当前: {trials_arr.shape}")
    if times_arr.ndim != 1 or times_arr.shape[0] != trials_arr.shape[1]:
        raise ValueError(
            f"times 长度 ({times_arr.shape}) 必须与 trials 时间维度 ({trials_arr.shape[1]}) 一致"
        )

    mask = (times_arr >= baseline[0]) & (times_arr < baseline[1])
    if not np.any(mask):
        raise ValueError(f"基线时间窗 {baseline} 内无有效采样点")

    energy = np.mean(trials_arr**2, axis=0)
    base_energy = float(np.mean(energy[mask]))
    if base_energy <= eps:
        return np.zeros_like(energy)
    return 100.0 * (energy - base_energy) / base_energy


def main() -> None:
    """命令行入口：生成 C3/C4 对侧 ERD/ERS 对比图与静息段 PSD 频谱图。"""
    ap = argparse.ArgumentParser(description="运动想象脑电 ERD/ERS 与静息频谱探索性分析脚本")
    ap.add_argument("--gdf", default="data_src/Train/A01T.gdf", help="输入 GDF 文件路径")
    ap.add_argument("--out-dir", default="results", help="图表输出目录")
    args = ap.parse_args()

    import mne

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_gdf(args.gdf, preload=True, verbose="error")
    eeg = raw.copy().pick("eeg")
    events, code = mne.events_from_annotations(eeg, verbose="error")
    id2orig = parse_annotation_id_map(code)
    cls_events = [e for e in events if id2orig.get(int(e[2])) in (769, 770, 771, 772)]
    if not cls_events:
        raise ValueError(f"{args.gdf} 中未找到 769–772 运动想象事件")

    codes = {769: 0, 770: 1, 771: 2, 772: 3}
    mapped = np.array([[e[0], 0, codes[id2orig[int(e[2])]]] for e in cls_events], dtype=np.int64)
    epochs = mne.Epochs(
        eeg,
        mapped,
        tmin=-1.0,
        tmax=4.5,
        baseline=None,
        preload=True,
        verbose="error",
        reject_by_annotation=False,
    )

    # 1) 左右手想象时 C3/C4 的 μ/β 能量（ERD/ERS）：想象时对侧感觉运动皮层去同步
    left = epochs["0"].copy().filter(8, 30, verbose="error")  # 0=左手
    right = epochs["1"].copy().filter(8, 30, verbose="error")  # 1=右手
    t = left.times
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    try:
        series = [(left, "左手想象", "#4c72b0"), (right, "右手想象", "#dd8452")]
        for ax, ch in zip(axes, ["EEG-C3", "EEG-C4"], strict=True):
            for data, name, color in series:
                x = data.copy().pick([ch]).get_data()[:, 0, :]
                erd = compute_erd_percentage(x, t, baseline=(-1.0, 0.0))
                ax.plot(t, erd, label=name, color=color)
            ax.axvline(0, color="#999", linestyle="--", linewidth=1)
            ax.axhline(0, color="#ccc", linewidth=0.8)
            ax.set_xlabel("相对提示出现的时间（秒）")
            ax.set_ylabel("相对基线能量变化（%）")
            ax.set_title(f"{ch.replace('EEG-', '')} 电极（8–30 Hz）")
            ax.legend()
        fig.suptitle("运动想象的 ERD/ERS：想象对侧肢体的皮层代表区能量下降更明显", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "erd_c3c4.png", dpi=150)
    finally:
        plt.close(fig)

    # 2) 静息段中央沟附近电极的频谱（对短记录做安全夹紧保护）
    max_t = float(eeg.times[-1])
    crop_start = 60.0 if max_t > 130.0 else 0.0
    crop_end = min(130.0, max_t)
    psd = eeg.copy().crop(crop_start, crop_end).compute_psd(fmin=1, fmax=45, verbose="error")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    try:
        for ch in ["EEG-C3", "EEG-Cz", "EEG-C4"]:
            d = psd.get_data(picks=[ch])[0]
            ax.plot(psd.freqs, 10 * np.log10(d + 1e-24), label=ch.replace("EEG-", ""))
        ax.set_xlabel("频率（Hz）")
        ax.set_ylabel("功率谱密度（dB）")
        ax.set_title("静息段中央沟附近电极的频谱：能量集中在 8–30 Hz（μ/β 节律）")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "psd.png", dpi=150)
    finally:
        plt.close(fig)
    print(f"图已保存到 {out}")


if __name__ == "__main__":
    main()
