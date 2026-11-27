"""三模型多数投票融合（CSP+LDA / ShallowConvNet / FBCSP），支持主协议与跨会话协议。

当至少两个模型预测一致时取多数票；当三个模型预测各不相同时，
偏向单模型准确率最高的平票决胜模型（默认为 FBCSP）。

用法:
    python src/vote_ensemble.py
    python src/vote_ensemble.py --in-dir outputs/final --out-dir outputs/final/vote
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

try:
    from src.metrics import accuracy_score_np, cohen_kappa_score_np
except ImportError:
    from metrics import accuracy_score_np, cohen_kappa_score_np


def majority_vote(
    preds_a: np.ndarray | Sequence[int],
    preds_b: np.ndarray | Sequence[int],
    preds_tiebreak: np.ndarray | Sequence[int],
) -> np.ndarray:
    """对三组硬分类预测执行逐 trial 多数投票。

    规则：
    - 若三个模型中至少有两个预测相同，采用多数类；
    - 若三个模型预测完全不同，采用 ``preds_tiebreak``（默认 FBCSP）作为决胜输出。

    参数:
        preds_a: 模型 A（CSP+LDA）的预测标签一维数组。
        preds_b: 模型 B（ShallowConvNet）的预测标签一维数组。
        preds_tiebreak: 平票决胜模型（FBCSP）的预测标签一维数组。

    返回:
        融合后的预测标签一维数组（``int64``）。
    """
    a = np.asarray(preds_a)
    b = np.asarray(preds_b)
    c = np.asarray(preds_tiebreak)
    if not (a.ndim == b.ndim == c.ndim == 1):
        raise ValueError(f"三组预测必须均为一维数组，当前形状: {a.shape}, {b.shape}, {c.shape}")
    if not (a.shape == b.shape == c.shape):
        raise ValueError(f"三组预测长度不一致: {a.shape[0]}, {b.shape[0]}, {c.shape[0]}")
    if a.size == 0:
        raise ValueError("预测数组不能为空")

    fused: list[int] = []
    for pa, pb, pc in zip(a.tolist(), b.tolist(), c.tolist(), strict=True):
        top_label, top_count = Counter((pa, pb, pc)).most_common(1)[0]
        fused.append(int(top_label) if top_count >= 2 else int(pc))
    return np.array(fused, dtype=np.int64)


def load_predictions(
    in_dir: Path,
    model_name: str,
    prefix: str = "pooled",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从指定模型目录读取预测结果数组并安全关闭文件句柄。

    参数:
        in_dir: 各模型输出结果的父目录（如 ``outputs/final``）。
        model_name: 模型子目录名称（``csp_lda`` / ``shallow`` / ``fbcsp``）。
        prefix: 协议前缀，``"pooled"`` 或 ``"cross"``。

    返回:
        ``(pred, y, sub)`` 三元组。
    """
    npz_path = in_dir / model_name / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"未找到模型预测文件: {npz_path}")
    with np.load(npz_path) as npz:
        return (
            np.array(npz[f"{prefix}_pred"]),
            np.array(npz[f"{prefix}_y"]),
            np.array(npz[f"{prefix}_sub"]),
        )


def load(
    name: str,
    in_dir: Path | str = "outputs/final",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """向后兼容的预测读取包装函数（默认读取主协议 ``pooled``）。"""
    return load_predictions(Path(in_dir), name, prefix="pooled")


def _validate_alignment(
    ref_y: np.ndarray,
    ref_sub: np.ndarray,
    others: list[tuple[str, np.ndarray, np.ndarray]],
    protocol: str,
) -> None:
    """校验多模型预测文件的真实标签与受试者顺序严格对齐，防止静默错位。"""
    for name, other_y, other_sub in others:
        if ref_y.shape != other_y.shape or not np.array_equal(ref_y, other_y):
            raise ValueError(
                f"[{protocol}] 模型 {name} 的真实标签 y 与基线不一致，无法安全投票"
            )
        if ref_sub.shape != other_sub.shape or not np.array_equal(ref_sub, other_sub):
            raise ValueError(
                f"[{protocol}] 模型 {name} 的受试者顺序 sub 与基线不一致，无法安全投票"
            )


def evaluate_by_subject(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
) -> tuple[list[tuple[str, float, float]], float, float]:
    """按受试者聚合计算准确率与 Cohen's kappa，并返回总体指标。"""
    subs = sorted(set(subjects.tolist()))
    rows: list[tuple[str, float, float]] = []
    for s in subs:
        mask = subjects == s
        rows.append(
            (
                str(s),
                accuracy_score_np(y_true[mask], y_pred[mask]),
                cohen_kappa_score_np(y_true[mask], y_pred[mask]),
            )
        )
    overall_acc = accuracy_score_np(y_true, y_pred)
    overall_kappa = cohen_kappa_score_np(y_true, y_pred)
    return rows, overall_acc, overall_kappa


def run_vote_ensemble(in_dir: Path, out_dir: Path) -> dict[str, float]:
    """执行主协议与跨会话协议的三模型投票融合并保存 CSV 与 NPZ 结果。

    参数:
        in_dir: 包含 ``csp_lda``、``shallow``、``fbcsp`` 子目录的输入路径。
        out_dir: 投票融合结果输出目录。

    返回:
        包含两个协议总体准确率与 kappa 的指标字典。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 主协议（合并五折）多数投票
    csp_pred, y, sub = load_predictions(in_dir, "csp_lda", prefix="pooled")
    sh_pred, sh_y, sh_sub = load_predictions(in_dir, "shallow", prefix="pooled")
    fb_pred, fb_y, fb_sub = load_predictions(in_dir, "fbcsp", prefix="pooled")
    _validate_alignment(y, sub, [("shallow", sh_y, sh_sub), ("fbcsp", fb_y, fb_sub)], "pooled")

    pred = majority_vote(csp_pred, sh_pred, fb_pred)
    rows, acc, kappa = evaluate_by_subject(y, pred, sub)
    for s, s_acc, _ in rows:
        print(f"{s}: 投票 {s_acc:.4f}")
    print("")
    print(f"=== 投票融合 / 合并五折（{len(rows)} 人）===")
    print(f"准确率 {acc:.4f} | kappa {kappa:.4f}")
    with open(out_dir / "pooled_cv10.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["受试者", "准确率", "kappa"])
        writer.writerows(rows)
        writer.writerow(["均值", f"{acc:.4f}", f"{kappa:.4f}"])

    # 2) 跨会话协议（T → E）多数投票
    csp_cpred, cy, csub = load_predictions(in_dir, "csp_lda", prefix="cross")
    sh_cpred, sh_cy, sh_csub = load_predictions(in_dir, "shallow", prefix="cross")
    fb_cpred, fb_cy, fb_csub = load_predictions(in_dir, "fbcsp", prefix="cross")
    _validate_alignment(
        cy,
        csub,
        [("shallow", sh_cy, sh_csub), ("fbcsp", fb_cy, fb_csub)],
        "cross_session",
    )

    cpred = majority_vote(csp_cpred, sh_cpred, fb_cpred)
    crows, cacc, ckappa = evaluate_by_subject(cy, cpred, csub)
    with open(out_dir / "cross_session.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["受试者", "准确率", "kappa"])
        writer.writerows(crows)
        writer.writerow(["均值", f"{cacc:.4f}", f"{ckappa:.4f}"])
    print(f"跨会话投票: acc {cacc:.4f} kappa {ckappa:.4f}")

    # 同步保存两协议预测明细，格式与其余单模型保持一致
    np.savez(
        out_dir / "predictions.npz",
        pooled_pred=pred,
        pooled_y=y,
        pooled_sub=sub,
        cross_pred=cpred,
        cross_y=cy,
        cross_sub=csub,
    )
    return {
        "pooled_acc": acc,
        "pooled_kappa": kappa,
        "cross_acc": cacc,
        "cross_kappa": ckappa,
    }


def main() -> None:
    """命令行入口。"""
    ap = argparse.ArgumentParser(description="三模型多数投票融合脚本")
    ap.add_argument("--in-dir", default="outputs/final", help="单模型预测结果父目录")
    ap.add_argument("--out-dir", default="outputs/final/vote", help="投票结果输出目录")
    args = ap.parse_args()
    run_vote_ensemble(Path(args.in_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
