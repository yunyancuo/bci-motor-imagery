"""评估与图表生成：CSP+LDA（基线）/ FBCSP（改进）/ 三模型投票（融合）双协议汇总。

用法:
    python src/evaluate.py  # 默认读取 outputs/final/ 并输出到 results/
    python src/evaluate.py --in-dir outputs/final --out-dir results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import src.cn_font  # noqa: F401  注册中文字体
    from src.metrics import CLASS_NAMES, accuracy_score_np, confusion_matrix_np
except ImportError:
    import cn_font  # noqa: F401  注册中文字体
    from metrics import CLASS_NAMES, accuracy_score_np, confusion_matrix_np

METHODS: list[tuple[str, str]] = [
    ("csp_lda", "CSP+LDA（基线）"),
    ("fbcsp", "FBCSP（改进）"),
    ("vote", "三模型投票（融合）"),
]


def load(
    name: str,
    in_dir: Path | str = "outputs/final",
) -> tuple[dict[str, float], dict[str, float], dict[str, np.ndarray], float, float | None]:
    """安全加载单个模型的 CSV 指标与 NPZ 预测结果。

    参数:
        name: 模型子目录名（``csp_lda`` / ``fbcsp`` / ``vote``）。
        in_dir: 模型结果父目录，默认 ``outputs/final``。

    返回:
        ``(pooled_acc, pooled_kappa, predictions_dict, cross_acc_mean, cross_kappa_mean)`` 五元组。
    """
    model_dir = Path(in_dir) / name
    with open(model_dir / "pooled_cv10.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["受试者"] != "均值"]
    if not rows:
        raise ValueError(f"{model_dir / 'pooled_cv10.csv'} 中缺少有效受试者记录")

    acc = {r["受试者"]: float(r["准确率"]) for r in rows}
    kap = {r["受试者"]: float(r["kappa"]) for r in rows}

    with np.load(model_dir / "predictions.npz") as npz:
        preds_dict = {k: np.array(npz[k]) for k in npz.files}

    with open(model_dir / "cross_session.csv", encoding="utf-8-sig") as f:
        cross = [r for r in csv.DictReader(f) if r["受试者"] != "均值"]
    if not cross:
        raise ValueError(f"{model_dir / 'cross_session.csv'} 中缺少有效跨会话记录")

    cacc = float(np.mean([float(r["准确率"]) for r in cross]))
    ckap_vals = [float(r["kappa"]) for r in cross if r.get("kappa", "").strip() != ""]
    ckap = float(np.mean(ckap_vals)) if ckap_vals else None
    return acc, kap, preds_dict, cacc, ckap


def generate_evaluation(in_dir: Path, out_dir: Path) -> list[str]:
    """生成逐受试者对比柱状图、混淆矩阵图、汇总表与逐 trial 预测明细。

    参数:
        in_dir: 模型输出结果目录（包含 ``csp_lda``、``fbcsp``、``vote``）。
        out_dir: 图表与 CSV 报告输出目录。

    返回:
        写入 ``summary.csv`` 的文本行列表。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {m: load(m, in_dir=in_dir) for m, _ in METHODS}
    subs = sorted(data["csp_lda"][0].keys())
    for m, _ in METHODS:
        missing = [s for s in subs if s not in data[m][0]]
        if missing:
            raise ValueError(f"模型 {m} 缺少受试者记录: {missing}")

    # 1) 逐受试者柱状图（主协议）
    x = np.arange(len(subs))
    bar_width = 0.27
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    try:
        for idx, ((m, label), color) in enumerate(zip(METHODS, colors, strict=True)):
            ax.bar(
                x + (idx - 1) * bar_width,
                [data[m][0][s] for s in subs],
                bar_width,
                label=label,
                color=color,
            )
        ax.axhline(0.25, color="#888", linestyle="--", linewidth=1)
        ax.text(max(len(subs) - 0.6, 0.0), 0.265, "随机水平 25%", fontsize=9, color="#666")
        ax.set_xticks(x, subs)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("合并五折交叉验证准确率")
        ax.set_title("逐受试者准确率（两会话合并，五折交叉验证）")
        ax.legend(
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            framealpha=0.9,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "per_subject_acc.png", dpi=150)
    finally:
        plt.close(fig)

    # 2) 混淆矩阵（融合模型，主协议聚合）
    vote_preds = data["vote"][2]
    cm = confusion_matrix_np(
        vote_preds["pooled_y"],
        vote_preds["pooled_pred"],
        n_classes=len(CLASS_NAMES),
        normalize="true",
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    try:
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > 0.5 else "black",
                    fontsize=10,
                )
        acc = accuracy_score_np(vote_preds["pooled_y"], vote_preds["pooled_pred"])
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(f"三模型投票融合的混淆矩阵（行归一化），总体准确率 {acc:.3f}")
        fig.colorbar(im, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    finally:
        plt.close(fig)

    # 3) 汇总表
    lines = ["方法,协议,准确率均值,kappa均值"]
    for m, label in METHODS:
        accs = np.array([data[m][0][s] for s in subs], dtype=np.float64)
        kaps = np.array([data[m][1][s] for s in subs], dtype=np.float64)
        ckap = data[m][4]
        ckap_str = f"{ckap:.4f}" if ckap is not None else ""
        lines.append(f"{label},合并五折,{accs.mean():.4f},{kaps.mean():.4f}")
        lines.append(f"{label},跨会话,{data[m][3]:.4f},{ckap_str}")
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print("\n".join(lines))

    # 4) 逐 trial 明细（融合模型，主协议）
    with open(out_dir / "vote_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["受试者", "trial 序号", "真实类别", "预测类别", "是否正确"])
        for i, (s, y_val, p_val) in enumerate(
            zip(
                vote_preds["pooled_sub"],
                vote_preds["pooled_y"],
                vote_preds["pooled_pred"],
                strict=True,
            )
        ):
            writer.writerow(
                [
                    s,
                    i,
                    CLASS_NAMES[int(y_val)],
                    CLASS_NAMES[int(p_val)],
                    "正确" if int(y_val) == int(p_val) else "错误",
                ]
            )
    print(f"图与明细已保存到 {out_dir}")
    return lines


def main() -> None:
    """命令行入口。"""
    ap = argparse.ArgumentParser(description="生成双协议汇总表、对比图表与逐 trial 预测明细")
    ap.add_argument("--in-dir", default="outputs/final", help="模型输出根目录")
    ap.add_argument("--out-dir", default="results", help="评估图表与汇总文件输出目录")
    args = ap.parse_args()
    generate_evaluation(Path(args.in_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
