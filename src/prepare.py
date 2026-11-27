"""BCI Competition IV 2a 数据准备：安全解包 GDF + 真标签 → 每受试者 npz 文件。

数据集：9 名受试者 × 2 个会话（T=训练，E=测试），22 导 EEG，250 Hz，
四类运动想象（左手/右手/双脚/舌头），每会话 288 个 trial（每类 72 个）。
T 会话的类别直接编码在 GDF 事件中（769–772）；E 会话 GDF 中为“未知”（783），
真实类别位于官方提供的 A0xE.mat 文件中，按 trial 顺序一一对应。

用法:
    python src/prepare.py --src-dir data_src --out-dir data/bci2a
输出:
    data/bci2a/A01T.npz ... A09E.npz   # X: (n, 22, fs*t) float32, y: (n,) 0-3
"""

from __future__ import annotations

import argparse
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

FS: int = 250
TMIN: float = 0.5
TMAX: float = 4.0  # 相对提示出现的运动想象时间窗（秒）
CLASS_CODES: dict[int, int] = {769: 0, 770: 1, 771: 2, 772: 3}  # 左手/右手/双脚/舌头
CLASS_NAMES: list[str] = ["左手", "右手", "双脚", "舌头"]


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """安全解压 ZIP 压缩包，防御 Zip Slip 路径穿越攻击。

    参数:
        zip_path: 待解压的 ``.zip`` 文件路径。
        dest_dir: 目标解压目录。

    异常:
        ValueError: 当压缩包内存在指向目标目录之外的非法成员路径时抛出。
    """
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target_path = (dest_resolved / member.filename).resolve()
            if not target_path.is_relative_to(dest_resolved):
                raise ValueError(f"检测到非法压缩包路径（Zip Slip 风险）: {member.filename!r}")
        zf.extractall(dest_resolved)


def load_mat_labels(path: Path | str) -> np.ndarray:
    """读取官方 ``A0xE.mat`` 中的 ``classlabel``（1–4 编码）并转换为 0–3 整数标签。

    参数:
        path: ``.mat`` 文件路径。

    返回:
        形状为 ``(n_trials,)``、类型为 ``int64`` 的标签数组（取值 0–3）。
    """
    from scipy.io import loadmat

    mat = loadmat(str(path))
    if "classlabel" not in mat:
        raise ValueError(f"{path} 中未找到 classlabel 字段")
    raw_labels = np.asarray(mat["classlabel"]).ravel()
    if raw_labels.size == 0:
        raise ValueError(f"{path} 中 classlabel 为空")
    labels = np.array([int(val) - 1 for val in raw_labels], dtype=np.int64)
    if np.any((labels < 0) | (labels >= len(CLASS_NAMES))):
        raise ValueError(f"{path} 包含越界类别标签，合法编码应为 1–{len(CLASS_NAMES)}")
    return labels


def parse_annotation_id_map(code_map: dict[str, int]) -> dict[int, int]:
    """将 MNE 注释编码字典安全反转为 ``event_id -> 原始数字编码``，跳过非数字注释。"""
    id2orig: dict[int, int] = {}
    for key, val in code_map.items():
        try:
            id2orig[int(val)] = int(str(key).strip())
        except ValueError:
            continue
    return id2orig


def extract_epochs(
    gdf_path: Path | str,
    true_labels: Sequence[int] | np.ndarray | None = None,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    tmin: float = TMIN,
    tmax: float = TMAX,
) -> tuple[np.ndarray, np.ndarray]:
    """从 GDF 文件提取运动想象 EEG epoch。

    参数:
        gdf_path: 原始 GDF 文件路径。
        true_labels: 非 ``None`` 时（E 会话）按 trial 顺序赋予真实标签。
        l_freq: 带通下截止频率（Hz），默认 8.0。
        h_freq: 带通上截止频率（Hz），默认 30.0。
        tmin: 相对事件提示的起始时间（秒），默认 0.5。
        tmax: 相对事件提示的结束时间（秒），默认 4.0。

    返回:
        ``(X, y)`` 元组，其中 ``X`` 形状为 ``(n_trials, 22, n_times)``（``float32``），
        ``y`` 形状为 ``(n_trials,)``（``int64``）。
    """
    import mne

    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="error")
    eog = [ch for ch in raw.ch_names if ch.startswith("EOG")]
    if eog:
        raw.drop_channels(eog)  # 剔除 3 导眼电，仅保留 22 导脑电
    raw.filter(l_freq, h_freq, verbose="error")  # μ/β 频段（ERD/ERS 所在）

    events, code = mne.events_from_annotations(raw, verbose="error")
    id2orig = parse_annotation_id_map(code)
    picks = []
    labels: list[int | None] = []
    for ev in events:
        orig = id2orig.get(int(ev[2]))
        if orig in CLASS_CODES:
            picks.append(ev)
            labels.append(CLASS_CODES[orig])
        elif orig == 783:  # E 会话的“未知”提示事件
            picks.append(ev)
            labels.append(None)

    if not picks:
        raise ValueError(f"{gdf_path} 中未提取到有效的运动想象事件（769–772 或 783）")

    picks_arr = np.asarray(picks, dtype=np.int64)
    if true_labels is not None:
        true_arr = [int(v) for v in np.asarray(true_labels).ravel()]
        if len(labels) != len(true_arr):
            raise ValueError(
                f"{gdf_path}: 事件 trial 数 ({len(labels)}) 与真标签数 ({len(true_arr)}) 不一致"
            )
        final_labels = true_arr
    else:
        if any(lab is None for lab in labels):
            raise ValueError(f"{gdf_path} 包含未标注事件 783，但未提供 true_labels")
        final_labels = [int(lab) for lab in labels if lab is not None]

    epochs = mne.Epochs(
        raw,
        picks_arr,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="error",
        reject_by_annotation=False,
    )
    X = epochs.get_data().astype(np.float32)
    y = np.array(final_labels, dtype=np.int64)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{gdf_path}: 提取的 epoch 数 ({X.shape[0]}) 与标签数 ({y.shape[0]}) 不一致"
        )
    return X, y


def main() -> None:
    """命令行入口：批量解包、滤波并导出 9 名受试者双会话 npz 数据。"""
    ap = argparse.ArgumentParser(description="BCI Competition IV 2a GDF 数据预处理脚本")
    ap.add_argument("--src-dir", default="data_src", help="原始 GDF/ZIP 与标签文件目录")
    ap.add_argument("--out-dir", default="data/bci2a", help="预处理 npz 输出目录")
    ap.add_argument("--l-freq", type=float, default=8.0, help="带通下截止频率 Hz（默认 8.0）")
    ap.add_argument("--h-freq", type=float, default=30.0, help="带通上截止频率 Hz（默认 30.0）")
    ap.add_argument("--tmin", type=float, default=TMIN, help="Epoch 相对起点秒数（默认 0.5）")
    ap.add_argument("--tmax", type=float, default=TMAX, help="Epoch 相对终点秒数（默认 4.0）")
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for z in sorted(src.glob("*.zip")):
        print(f"解包 {z.name}")
        safe_extract_zip(z, src)

    lab_dir: Path | None = None
    for cand in [src, src / "true_labels", src / "TrueLabels"]:
        if list(cand.glob("A01E.mat")):
            lab_dir = cand
            break
    if lab_dir is None:
        raise FileNotFoundError(f"在 {src} 及其常见子目录中未找到 A0xE.mat 真标签文件")

    for sub in range(1, 10):
        for sess in ["T", "E"]:
            gdfs = sorted(src.rglob(f"A{sub:02d}{sess}.gdf"))
            if not gdfs:
                print(f"缺 A{sub:02d}{sess}.gdf，跳过")
                continue
            tl = load_mat_labels(lab_dir / f"A{sub:02d}{sess}.mat") if sess == "E" else None
            X, y = extract_epochs(
                gdfs[0],
                tl,
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                tmin=args.tmin,
                tmax=args.tmax,
            )
            np.savez_compressed(out / f"A{sub:02d}{sess}.npz", X=X, y=y)
            print(f"A{sub:02d}{sess}: X{X.shape} 类别分布 {np.bincount(y)}")


if __name__ == "__main__":
    main()
