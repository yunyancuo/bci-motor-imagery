"""统一实验脚本：基线 CSP+LDA、改进 ShallowConvNet 及概率集成，双协议评估。

协议一（主协议）：受试者两会话合并后的五折分层交叉验证（无跨受试者、无跨折泄漏）；
协议二（附加协议）：跨会话泛化——T 会话训练、E 会话测试（隔天采集，存在非平稳偏移）。

用法:
    python src/run_experiment.py --model csp_lda
    python src/run_experiment.py --model shallow --epochs 250
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

try:
    from src.metrics import (
        CLASS_NAMES,
        accuracy_score_np,
        apply_zscore,
        cohen_kappa_score_np,
        stratified_folds,
        zscore_stats,
    )
except ImportError:
    from metrics import (
        CLASS_NAMES,
        accuracy_score_np,
        apply_zscore,
        cohen_kappa_score_np,
        stratified_folds,
        zscore_stats,
    )

__all__ = [
    "CLASS_NAMES",
    "FOLDS",
    "WIN",
    "N_CROPS",
    "zscore_stats",
    "stratified_folds",
    "run_csp",
    "run_shallow",
    "main",
]

FOLDS: int = 5
WIN: int = 750  # 训练/测试滑窗长度：3 秒（从 876 采样点中滑窗裁剪）
N_CROPS: int = 3  # 每个训练 trial 每轮随机裁剪的增强窗口数


def run_csp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    n_components: int = 6,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """运行基线 CSP + LDA 管道（所有统计量严格仅在训练集上拟合）。

    参数:
        Xtr: 训练集 EEG 信号 ``(N_tr, C, T)``。
        ytr: 训练集类别标签 ``(N_tr,)``。
        Xte: 测试集 EEG 信号 ``(N_te, C, T)``。
        yte: 测试集类别标签 ``(N_te,)``。
        n_components: CSP 空间滤波器数量，默认 6。

    返回:
        ``(accuracy, kappa, pred, prob)`` 四元组。
    """
    import mne
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline

    csp = mne.decoding.CSP(
        n_components=n_components,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False,
    )
    clf = Pipeline([("csp", csp), ("lda", LinearDiscriminantAnalysis())])
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


def run_shallow(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    epochs: int = 120,
    seed: int = 42,
    device: Any = None,
    win: int = WIN,
    n_crops: int = N_CROPS,
    n_filters: int = 60,
    batch_size: int = 64,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """训练并评估 ShallowConvNet（含训练集滑窗增强与测试集五窗概率平均）。

    参数:
        Xtr: 训练集 EEG 信号 ``(N_tr, C, T)``。
        ytr: 训练集类别标签 ``(N_tr,)``。
        Xte: 测试集 EEG 信号 ``(N_te, C, T)``。
        yte: 测试集类别标签 ``(N_te,)``。
        epochs: 训练轮数，默认 120。
        seed: 随机种子，默认 42。
        device: PyTorch 计算设备；为 ``None`` 时自动优先使用 CUDA。
        win: 滑窗长度（采样点），默认 750。
        n_crops: 每轮每条训练样本随机裁剪次数，默认 3。
        n_filters: ShallowConvNet 滤波器数，默认 60。
        batch_size: 小批量大小，默认 64。

    返回:
        ``(accuracy, kappa, pred, prob)`` 四元组。
    """
    import torch

    try:
        from src.models import ShallowConvNet
    except ImportError:
        from models import ShallowConvNet

    xtr_arr = np.asarray(Xtr, dtype=np.float32)
    xte_arr = np.asarray(Xte, dtype=np.float32)
    ytr_arr = np.asarray(ytr, dtype=np.int64)
    yte_arr = np.asarray(yte, dtype=np.int64)

    if xtr_arr.ndim != 3 or xte_arr.ndim != 3:
        raise ValueError(f"输入 EEG 必须为 3D 张量，当前: Xtr={xtr_arr.shape}, Xte={xte_arr.shape}")
    if xtr_arr.shape[1:] != xte_arr.shape[1:]:
        raise ValueError(
            f"训练与测试集的通道或时间维度不一致: {xtr_arr.shape[1:]} vs {xte_arr.shape[1:]}"
        )
    if epochs < 1:
        raise ValueError(f"epochs 必须 >= 1，当前为 {epochs}")

    n_samples, n_ch, n_times = xtr_arr.shape
    if n_times < win:
        raise ValueError(f"输入时间长度 ({n_times}) 小于滑窗长度 win ({win})")

    n_classes = max(int(ytr_arr.max()) + 1, len(CLASS_NAMES))
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 标准化统计量严格仅在训练集上计算，防止测试集分布信息泄漏
    mu, sd = zscore_stats(xtr_arr)
    xg = torch.tensor(apply_zscore(xtr_arr, mu, sd), dtype=torch.float32, device=resolved_device)
    yg = torch.tensor(ytr_arr, dtype=torch.long, device=resolved_device)
    xe = torch.tensor(apply_zscore(xte_arr, mu, sd), dtype=torch.float32, device=resolved_device)

    model = ShallowConvNet(
        n_ch=n_ch,
        n_classes=n_classes,
        n_times=win,
        n_filters=n_filters,
    ).to(resolved_device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = torch.nn.CrossEntropyLoss()
    offs = torch.arange(win, device=resolved_device)

    for _ in range(epochs):
        model.train()
        starts = torch.randint(0, n_times - win + 1, (n_samples * n_crops,), device=resolved_device)
        trial = torch.arange(n_samples, device=resolved_device).repeat_interleave(n_crops)
        lab = yg.repeat_interleave(n_crops)
        perm = torch.randperm(n_samples * n_crops, device=resolved_device)
        trial, lab, st = trial[perm], lab[perm], starts[perm]
        for i in range(0, len(trial), batch_size):
            idx = trial[i : i + batch_size]
            ss = st[i : i + batch_size]
            gather_idx = (ss[:, None] + offs[None, :]).unsqueeze(1).expand(-1, n_ch, -1)
            xb = xg[idx].gather(2, gather_idx)
            loss = crit(model(xb), lab[i : i + batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        eval_starts = np.linspace(0, n_times - win, 5).astype(int)
        probs = [torch.softmax(model(xe[:, :, s : s + win]), dim=1) for s in eval_starts]
    prob = torch.stack(probs).mean(dim=0).cpu().numpy()
    pred = prob.argmax(axis=1)
    return (
        accuracy_score_np(yte_arr, pred),
        cohen_kappa_score_np(yte_arr, pred),
        pred,
        prob,
    )


def main() -> None:
    """命令行入口：按指定模型执行全部受试者的双协议实验。"""
    ap = argparse.ArgumentParser(description="BCI Competition IV 2a 统一模型训练与双协议评估脚本")
    ap.add_argument("--data-dir", default="data/bci2a", help="预处理 npz 数据目录")
    ap.add_argument(
        "--model",
        choices=["csp_lda", "shallow", "ensemble"],
        required=True,
        help="待评估模型类型",
    )
    ap.add_argument("--epochs", type=int, default=250, help="ShallowConvNet 训练轮数（默认 250）")
    ap.add_argument("--folds", type=int, default=FOLDS, help="主协议分层交叉验证折数（默认 5）")
    ap.add_argument("--seed", type=int, default=42, help="分层划分随机种子（默认 42）")
    ap.add_argument("--out-dir", default=None, help="实验输出目录，默认 outputs/<model>")
    args = ap.parse_args()

    out = Path(args.out_dir or f"outputs/{args.model}")
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    def run_one(
        model_name: str,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_te: np.ndarray,
        y_te: np.ndarray,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        if model_name == "csp_lda":
            return run_csp(x_tr, y_tr, x_te, y_te)
        if model_name == "shallow":
            return run_shallow(x_tr, y_tr, x_te, y_te, epochs=args.epochs, seed=args.seed)
        _, _, _, p1 = run_csp(x_tr, y_tr, x_te, y_te)
        _, _, _, p2 = run_shallow(x_tr, y_tr, x_te, y_te, epochs=args.epochs, seed=args.seed)
        _, _, _, p3 = run_shallow(x_tr, y_tr, x_te, y_te, epochs=args.epochs, seed=7)
        prob = (p1 + p2 + p3) / 3.0
        pred = prob.argmax(axis=1)
        y_te_arr = np.asarray(y_te)
        return (
            accuracy_score_np(y_te_arr, pred),
            cohen_kappa_score_np(y_te_arr, pred),
            pred,
            prob,
        )

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

        # 主协议：受试者两会话合并，五折分层交叉验证（无跨受试者、无跨折泄漏）
        X = np.concatenate([x_t, x_e])
        y = np.concatenate([y_t, y_e])
        folds = stratified_folds(y, k=args.folds, seed=args.seed)
        accs: list[float] = []
        ks: list[float] = []
        for fold_idx, te_idx in enumerate(folds):
            tr_idx = np.concatenate([f for j, f in enumerate(folds) if j != fold_idx])
            acc, kap, pred, _ = run_one(args.model, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
            accs.append(acc)
            ks.append(kap)
            p_preds.append(pred)
            p_ys.append(y[te_idx])
            p_subs.append([f"A{sub:02d}"] * len(te_idx))
        pooled_rows.append((f"A{sub:02d}", float(np.mean(accs)), float(np.mean(ks))))

        # 附加协议：T 会话训练 → E 会话测试（跨天泛化）
        acc, kap, pred, _ = run_one(args.model, x_t, y_t, x_e, y_e)
        cross_rows.append((f"A{sub:02d}", acc, kap))
        c_preds.append(pred)
        c_ys.append(y_e)
        c_subs.append([f"A{sub:02d}"] * len(y_e))
        print(f"A{sub:02d}: 合并五折 {np.mean(accs):.4f} | 跨会话 {acc:.4f}", flush=True)

    if not pooled_rows:
        raise FileNotFoundError(f"在 {data_dir} 下未找到任何成对的 A0xT.npz / A0xE.npz 数据文件")

    # 输出文件名沿用 pooled_cv10.csv 以保持与下游评估脚本接口兼容
    for name, rows in [("pooled_cv10", pooled_rows), ("cross_session", cross_rows)]:
        accs_arr = np.array([r[1] for r in rows], dtype=np.float64)
        ks_arr = np.array([r[2] for r in rows], dtype=np.float64)
        print("")
        print(f"=== {args.model} / {name}（{len(rows)} 人）===")
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
