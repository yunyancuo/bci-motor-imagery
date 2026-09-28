"""EDA：运动想象引起的 ERD/ERS 示意（C3/C4 左右手对比）+ 频谱图（报告用）。

注意：该 GDF 的通道名带 "EEG-" 前缀（如 EEG-C3），事件条件名为类别序号 0-3。
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cn_font  # noqa: F401  注册中文字体
import matplotlib.pyplot as plt
import mne
import numpy as np

CLASS_NAMES = ["左手", "右手", "双脚", "舌头"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdf", default="data_src/Train/A01T.gdf")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_gdf(args.gdf, preload=True, verbose="error")
    eeg = raw.copy().pick_types(eeg=True)
    events, code = mne.events_from_annotations(eeg, verbose="error")
    id2orig = {v: int(k) for k, v in code.items()}
    cls_events = [e for e in events if id2orig[e[2]] in (769, 770, 771, 772)]
    codes = {769: 0, 770: 1, 771: 2, 772: 3}
    mapped = np.array([[e[0], 0, codes[id2orig[e[2]]]] for e in cls_events])
    epochs = mne.Epochs(eeg, mapped, tmin=-1.0, tmax=4.5, baseline=None,
                        preload=True, verbose="error", reject_by_annotation=False)

    # 1) 左右手想象时 C3/C4 的 μ/β 能量（ERD/ERS）：想象时对侧感觉运动皮层去同步
    left = epochs["0"].copy().filter(8, 30, verbose="error")   # 0=左手
    right = epochs["1"].copy().filter(8, 30, verbose="error")  # 1=右手
    t = left.times
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ch in zip(axes, ["EEG-C3", "EEG-C4"]):
        for data, name, color in [(left, "左手想象", "#4c72b0"), (right, "右手想象", "#dd8452")]:
            x = data.copy().pick([ch]).get_data()[:, 0, :]
            energy = (x ** 2).mean(0)
            base = energy[(t >= -1) & (t < 0)].mean()
            erd = 100 * (energy - base) / base
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
    plt.close(fig)

    # 2) 静息段中央沟附近电极的频谱
    psd = eeg.copy().crop(60, 130).compute_psd(fmin=1, fmax=45, verbose="error")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for ch in ["EEG-C3", "EEG-Cz", "EEG-C4"]:
        d = psd.get_data(picks=[ch])[0]
        ax.plot(psd.freqs, 10 * np.log10(d + 1e-24), label=ch.replace("EEG-", ""))
    ax.set_xlabel("频率（Hz）")
    ax.set_ylabel("功率谱密度（dB）")
    ax.set_title("静息段中央沟附近电极的频谱：能量集中在 8–30 Hz（μ/β 节律）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "psd.png", dpi=150)
    plt.close(fig)
    print(f"图已保存到 {out}")


if __name__ == "__main__":
    main()
