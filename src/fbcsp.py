"""FBCSP（滤波器组 CSP + LDA）：BCI Competition IV 2a 竞赛冠军方法。

将 8–32 Hz 拆分为 6 个 4 Hz 子带，每个子带独立学习 4 个 CSP 空间滤波器，
拼接 24 维对数方差特征后交由 LDA 判别——不同受试者的 ERD/ERS 优势频段
存在个体差异，多子带滤波器组相比单一 8–30 Hz 频带具有更强鲁棒性。

用法:
    python src/fbcsp.py --out-dir outputs/final/fbcsp
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:  # 允许在未安装 scikit-learn 的轻量环境中复用纯 SciPy 滤波与特征逻辑

    class BaseEstimator:  # type: ignore[no-redef]
        """无 scikit-learn 环境下的轻量估计器基类占位。"""

    class TransformerMixin:  # type: ignore[no-redef]
        """无 scikit-learn 环境下的轻量转换器混入基类占位。"""

        def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
            return self.fit(X, y).transform(X)  # type: ignore[attr-defined]


try:
    from src.metrics import accuracy_score_np, cohen_kappa_score_np, stratified_folds
except ImportError:
    from metrics import accuracy_score_np, cohen_kappa_score_np, stratified_folds

BANDS: tuple[tuple[float, float], ...] = (
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 32),
)
FS: int = 250


def bandpass_filter(
    X: np.ndarray,
    lo: float,
    hi: float,
    fs: float = FS,
    order: int = 4,
) -> np.ndarray:
    """对三维脑电张量沿时间轴执行零相位 Butterworth 带通滤波。

    滤波按每个 trial 独立进行，不引入任何跨样本统计量（零数据泄漏）。

    参数:
        X: 形状为 ``(n_trials, n_channels, n_times)`` 的脑电数组。
        lo: 带通下截止频率（Hz）。
        hi: 带通上截止频率（Hz）。
        fs: 采样率（Hz），默认 250。
        order: Butterworth 滤波器阶数，默认 4。

    返回:
        滤波后的 ``float32`` 数组，与输入形状一致。
    """
    x_arr = np.asarray(X, dtype=np.float64)
    if x_arr.ndim != 3:
        raise ValueError(f"输入 X 必须为 3D (N, C, T)，当前形状: {x_arr.shape}")
    if fs <= 0:
        raise ValueError(f"采样率 fs 必须为正数，当前为 {fs}")
    nyquist = fs / 2.0
    if not (0 < lo < hi < nyquist):
        raise ValueError(
            f"子带范围必须满足 0 < lo < hi < fs/2 ({nyquist} Hz)，当前为 ({lo}, {hi})"
        )
    if order < 1:
        raise ValueError(f"滤波器阶数 order 必须 >= 1，当前为 {order}")

    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    # sosfiltfilt 默认 padlen = 3 * (2 * n_sections + 1 - min(...))，要求时间点数大于 padlen
    min_len = 3 * (2 * sos.shape[0] + 1)
    if x_arr.shape[2] <= min_len:
        raise ValueError(
            f"时间长度过短 ({x_arr.shape[2]})，零相位滤波至少需要超过 {min_len} 个采样点"
        )
    return sosfiltfilt(sos, x_arr, axis=2).astype(np.float32)


class _SimpleMultiClassCSP:
    """纯 NumPy/SciPy 正则化多类 CSP 实现（仅在未安装 MNE 时作为优雅回退后端）。"""

    def __init__(self, n_components: int = 4, reg: float = 1e-4) -> None:
        self.n_components = n_components
        self.reg = reg
        self.filters_: np.ndarray | None = None

    @staticmethod
    def _mean_cov(X: np.ndarray, reg: float) -> np.ndarray:
        """计算一组 trial 的迹归一化正则化平均协方差矩阵。"""
        n_ch = X.shape[1]
        covs = []
        for trial in X:
            c = trial @ trial.T
            trace = float(np.trace(c))
            if trace > 1e-12:
                c = c / trace
            covs.append(c)
        mean_c = np.mean(covs, axis=0)
        mean_c = (1.0 - reg) * mean_c + reg * (np.trace(mean_c) / n_ch) * np.eye(n_ch)
        return mean_c

    def fit(self, X: np.ndarray, y: np.ndarray) -> _SimpleMultiClassCSP:
        """基于一对多（OvR）广义特征分解拟合多类空间滤波器。"""
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        if classes.size < 2:
            raise ValueError("CSP 拟合至少需要 2 个类别")
        n_ch = x_arr.shape[1]
        if self.n_components > n_ch:
            raise ValueError(f"n_components ({self.n_components}) 不能大于通道数 ({n_ch})")

        # 对每个类别计算类内协方差与总体协方差的广义特征向量，再选最具区分度的滤波器
        eig_vecs_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []
        for cls in classes:
            c_pos = self._mean_cov(x_arr[y_arr == cls], self.reg)
            c_neg = self._mean_cov(x_arr[y_arr != cls], self.reg)
            c_sum = c_pos + c_neg + 1e-10 * np.eye(n_ch)
            eigvals, eigvecs = eigh(c_pos, c_sum)
            # 距 0.5 越远的特征值对应空间区分度越强
            scores = np.abs(eigvals - 0.5)
            eig_vecs_list.append(eigvecs.T)
            scores_list.append(scores)

        all_vecs = np.vstack(eig_vecs_list)
        all_scores = np.concatenate(scores_list)
        top_idx = np.argsort(all_scores)[::-1][: self.n_components]
        self.filters_ = all_vecs[top_idx]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """投影 EEG 信号并提取对数方差特征。"""
        if self.filters_ is None:
            raise RuntimeError("CSP 尚未拟合，请先调用 fit()")
        x_arr = np.asarray(X, dtype=np.float64)
        proj = np.einsum("kc,nct->nkt", self.filters_, x_arr)
        var = np.mean(proj * proj, axis=2)
        return np.log(np.clip(var, 1e-12, None))


def _default_csp_factory(n_components: int, reg: str | float) -> Any:
    """默认构造 MNE 的正则化 CSP；若未安装 MNE 则回退至纯 SciPy CSP。"""
    try:
        import mne

        return mne.decoding.CSP(
            n_components=n_components,
            reg=reg,
            log=True,
            norm_trace=False,
        )
    except ImportError:
        reg_val = 1e-4 if isinstance(reg, str) else float(reg)
        return _SimpleMultiClassCSP(n_components=n_components, reg=reg_val)


class FBCSP(TransformerMixin, BaseEstimator):
    """滤波器组共同空间模式（Filter Bank Common Spatial Patterns）特征提取器。"""

    def __init__(
        self,
        n_components: int = 4,
        bands: Sequence[tuple[float, float]] = BANDS,
        fs: float = FS,
        filter_order: int = 4,
        reg: str | float = "ledoit_wolf",
        csp_factory: Callable[[int, str | float], Any] | None = None,
    ) -> None:
        """初始化 FBCSP 参数。

        参数:
            n_components: 每个子带保留的 CSP 空间滤波器数量，默认 4。
            bands: 频段区间元组列表，默认 8–32 Hz 共 6 个 4 Hz 子带。
            fs: 采样率（Hz），默认 250。
            filter_order: Butterworth 滤波器阶数，默认 4。
            reg: CSP 协方差正则化策略，默认 ``"ledoit_wolf"``。
            csp_factory: 可选自定义单子带 CSP 构造工厂，便于测试或扩展。
        """
        self.n_components = n_components
        self.bands = bands
        self.fs = fs
        self.filter_order = filter_order
        self.reg = reg
        self.csp_factory = csp_factory

    def fit(self, X: np.ndarray, y: np.ndarray) -> FBCSP:
        """仅基于训练数据逐子带拟合独立 CSP 空间滤波器。"""
        if self.n_components < 1:
            raise ValueError(f"n_components 必须 >= 1，当前为 {self.n_components}")
        if not self.bands:
            raise ValueError("bands 频段列表不能为空")

        factory = self.csp_factory or _default_csp_factory
        self.csps_: list[Any] = []
        for lo, hi in self.bands:
            xb = self._band(X, lo, hi)
            csp = factory(self.n_components, self.reg)
            csp.fit(xb, y)
            self.csps_.append(csp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """将输入 EEG 信号映射为拼接后的多子带 CSP 对数方差特征矩阵。"""
        if not hasattr(self, "csps_"):
            raise RuntimeError("FBCSP 尚未拟合，请先调用 fit()")
        feats = [
            csp.transform(self._band(X, lo, hi))
            for csp, (lo, hi) in zip(self.csps_, self.bands, strict=True)
        ]
        return np.concatenate(feats, axis=1)

    def _band(self, X: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """对输入三维 EEG 张量执行单子带零相位带通滤波。"""
        return bandpass_filter(X, lo, hi, fs=self.fs, order=self.filter_order)


def run_fbcsp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    n_components: int = 4,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """在单次训练/测试划分上运行 FBCSP + LDA 管道。

    参数:
        Xtr: 训练集 EEG 信号 ``(N_tr, C, T)``。
        ytr: 训练集类别标签 ``(N_tr,)``。
        Xte: 测试集 EEG 信号 ``(N_te, C, T)``。
        yte: 测试集类别标签 ``(N_te,)``。
        n_components: 每个子带的 CSP 分量数，默认 4。

    返回:
        ``(accuracy, kappa, pred, prob)`` 四元组。
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline

    clf = Pipeline(
        [
            ("fbcsp", FBCSP(n_components=n_components)),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)
    pred = prob.argmax(axis=1)
    yte_arr = np.asarray(yte)
    return (
        accuracy_score_np(yte_arr, pred),
        cohen_kappa_score_np(yte_arr, pred),
        pred,
        prob,
    )


def main() -> None:
    """命令行入口：在 BCI Competition IV 2a 上执行合并五折与跨会话评估。"""
    ap = argparse.ArgumentParser(description="FBCSP + LDA 运动想象四分类评估脚本")
    ap.add_argument("--data-dir", default="data/bci2a", help="预处理 npz 数据目录")
    ap.add_argument("--out-dir", default="outputs/final/fbcsp", help="结果输出目录")
    ap.add_argument("--folds", type=int, default=5, help="主协议分层交叉验证折数（默认 5）")
    ap.add_argument("--seed", type=int, default=42, help="分层划分随机种子（默认 42）")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    pooled_rows: list[tuple[str, float, float]] = []
    cross_rows: list[tuple[str, float, float]] = []
    p_preds: list[np.ndarray] = []
    p_ys: list[np.ndarray] = []
    p_subs: list[list[str]] = []
    c_preds: list[np.ndarray] = []
    c_ys: list[np.ndarray] = []
    c_subs: list[list[str]] = []

    for sub in range(1, 10):
        t_path = data_dir / f"A{sub:02d}T.npz"
        e_path = data_dir / f"A{sub:02d}E.npz"
        if not t_path.exists() or not e_path.exists():
            continue
        with np.load(t_path) as train_npz, np.load(e_path) as eval_npz:
            x_t, y_t = train_npz["X"], train_npz["y"]
            x_e, y_e = eval_npz["X"], eval_npz["y"]

        X = np.concatenate([x_t, x_e])
        y = np.concatenate([y_t, y_e])
        folds = stratified_folds(y, k=args.folds, seed=args.seed)
        accs: list[float] = []
        ks: list[float] = []
        for fold_idx, te_idx in enumerate(folds):
            tr_idx = np.concatenate([f for j, f in enumerate(folds) if j != fold_idx])
            acc, kap, pred, _ = run_fbcsp(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
            accs.append(acc)
            ks.append(kap)
            p_preds.append(pred)
            p_ys.append(y[te_idx])
            p_subs.append([f"A{sub:02d}"] * len(te_idx))
        pooled_rows.append((f"A{sub:02d}", float(np.mean(accs)), float(np.mean(ks))))

        acc, kap, pred, _ = run_fbcsp(x_t, y_t, x_e, y_e)
        cross_rows.append((f"A{sub:02d}", acc, kap))
        c_preds.append(pred)
        c_ys.append(y_e)
        c_subs.append([f"A{sub:02d}"] * len(y_e))
        print(f"A{sub:02d}: 合并五折 {np.mean(accs):.4f} | 跨会话 {acc:.4f}", flush=True)

    if not pooled_rows:
        raise FileNotFoundError(f"在 {data_dir} 下未找到任何成对的 A0xT.npz / A0xE.npz 数据文件")

    # 输出文件名沿用 pooled_cv10.csv 以保持与下游 vote_ensemble.py / evaluate.py 兼容
    for name, rows in [("pooled_cv10", pooled_rows), ("cross_session", cross_rows)]:
        accs_arr = np.array([r[1] for r in rows], dtype=np.float64)
        ks_arr = np.array([r[2] for r in rows], dtype=np.float64)
        print("")
        print(f"=== fbcsp / {name}（{len(rows)} 人）===")
        print(
            f"准确率 {accs_arr.mean():.4f} ± {accs_arr.std():.4f} | "
            f"kappa {ks_arr.mean():.4f} ± {ks_arr.std():.4f}"
        )
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["受试者", "准确率", "kappa"])
            writer.writerows(rows)
            writer.writerow(["均值", f"{accs_arr.mean():.4f}", f"{ks_arr.mean():.4f}"])

    np.savez(
        out / "predictions.npz",
        pooled_pred=np.concatenate(p_preds),
        pooled_y=np.concatenate(p_ys),
        pooled_sub=np.concatenate(p_subs),
        cross_pred=np.concatenate(c_preds),
        cross_y=np.concatenate(c_ys),
        cross_sub=np.concatenate(c_subs),
    )
    print(f"保存到 {out}")


if __name__ == "__main__":
    main()
