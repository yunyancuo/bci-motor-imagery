"""BCI Competition IV 2a 数据准备：解包 gdf + 真标签 → 每受试者 npz。

数据集：9 名受试者 × 2 个会话（T=训练，E=测试），22 导 EEG，250 Hz，
四类运动想象（左手/右手/双脚/舌头），每会话 288 个 trial（每类 72 个）。
T 会话的类别直接编码在 gdf 事件里（769–772）；E 会话 gdf 里是"未知"(783)，
真实类别在官方提供的 A0xE.mat 里，按 trial 顺序对应。

用法：
    python src/prepare.py --src-dir data_src --out-dir data/bci2a
输出：
    data/bci2a/A01T.npz ... A09E.npz   # X: (n,22,fs*t) float32, y: (n,) 0-3
"""
import argparse
import glob
import zipfile
from pathlib import Path

import numpy as np

FS = 250
TMIN, TMAX = 0.5, 4.0  # 相对提示出现的运动想象时间窗（秒）
CLASS_CODES = {769: 0, 770: 1, 771: 2, 772: 3}  # 左手/右手/双脚/舌头
CLASS_NAMES = ["左手", "右手", "双脚", "舌头"]


def load_mat_labels(path):
    """A0xE.mat 里是 classlabel 向量（1-4 编码），转成 0-3。"""
    from scipy.io import loadmat
    m = loadmat(str(path))
    if "classlabel" not in m:
        raise ValueError(f"{path} 里没找到 classlabel")
    return np.array([int(t) - 1 for t in m["classlabel"].ravel()])


def extract_epochs(gdf_path, true_labels=None):
    """从 gdf 提取运动想象 epoch。true_labels 非 None 时（E 会话）按顺序赋真标签。"""
    import mne

    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="error")
    eog = [ch for ch in raw.ch_names if ch.startswith("EOG")]
    raw.drop_channels(eog)  # 22 导脑电，去掉 3 导眼电
    raw.filter(8.0, 30.0, verbose="error")  # μ/β 频段（ERD/ERS 所在）
    events, code = mne.events_from_annotations(raw, verbose="error")
    # code: 注释值→连续 id；反查原始编码
    id2orig = {v: int(k) for k, v in code.items()}
    picks = []
    labels = []
    for ev in events:
        orig = id2orig[ev[2]]
        if orig in CLASS_CODES:
            picks.append(ev)
            labels.append(CLASS_CODES[orig])
        elif orig == 783:  # E 会话的"未知"提示
            picks.append(ev)
            labels.append(None)
    picks = np.array(picks)
    if true_labels is not None:
        assert len(labels) == len(true_labels), f"{gdf_path}: {len(labels)} trials vs {len(true_labels)} labels"
        labels = list(true_labels)
    epochs = mne.Epochs(raw, picks, tmin=TMIN, tmax=TMAX, baseline=None,
                        preload=True, verbose="error", reject_by_annotation=False)
    X = epochs.get_data().astype(np.float32)  # (n, 22, n_times)
    y = np.array(labels, dtype=np.int64)
    assert X.shape[0] == y.shape[0]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="data_src")
    ap.add_argument("--out-dir", default="data/bci2a")
    args = ap.parse_args()
    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for z in src.glob("*.zip"):
        print(f"解包 {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(src)
    lab_dir = None
    for cand in [src, src / "true_labels", src / "TrueLabels"]:
        if list(cand.glob("A01E.mat")):
            lab_dir = cand
            break
    assert lab_dir, "没找到 A0xE.mat 真标签文件"

    for sub in range(1, 10):
        for sess in ["T", "E"]:
            gdfs = list(src.rglob(f"A{sub:02d}{sess}.gdf"))
            if not gdfs:
                print(f"缺 A{sub:02d}{sess}.gdf，跳过")
                continue
            tl = load_mat_labels(lab_dir / f"A{sub:02d}{sess}.mat") if sess == "E" else None
            X, y = extract_epochs(gdfs[0], tl)
            np.savez_compressed(out / f"A{sub:02d}{sess}.npz", X=X, y=y)
            print(f"A{sub:02d}{sess}: X{X.shape} 类别分布 {np.bincount(y)}")


if __name__ == "__main__":
    main()
