"""评估与图表：CSP+LDA（基线）/ FBCSP（改进）/ 三模型投票（融合），主协议对比。

用法：python src/evaluate.py   # 读取 outputs/final/ 下的结果
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cn_font  # noqa: F401  注册中文字体
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ["左手", "右手", "双脚", "舌头"]
METHODS = [
    ("csp_lda", "CSP+LDA（基线）"),
    ("fbcsp", "FBCSP（改进）"),
    ("vote", "三模型投票（融合）"),
]


def load(name):
    d = Path("outputs/final") / name
    rows = list(csv.DictReader(open(d / "pooled_cv10.csv", encoding="utf-8-sig")))
    rows = [r for r in rows if r["受试者"] != "均值"]
    acc = {r["受试者"]: float(r["准确率"]) for r in rows}
    kap = {r["受试者"]: float(r["kappa"]) for r in rows}
    npz = np.load(d / "predictions.npz")
    cross = list(csv.DictReader(open(d / "cross_session.csv", encoding="utf-8-sig")))
    cross = [r for r in cross if r["受试者"] != "均值"]
    cacc = float(np.mean([float(r["准确率"]) for r in cross]))
    return acc, kap, npz, cacc


def main():
    out = Path("results")
    out.mkdir(exist_ok=True)
    data = {m: load(m) for m, _ in METHODS}
    subs = sorted(data["csp_lda"][0].keys())

    # 1) 逐受试者柱状图（主协议）
    x = np.arange(len(subs))
    w = 0.27
    plt.figure(figsize=(8.5, 4.4))
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for (m, label), c in zip(METHODS, colors):
        plt.bar(x + (METHODS.index((m, label)) - 1) * w,
                [data[m][0][s] for s in subs], w, label=label, color=c)
    plt.axhline(0.25, color="#888", linestyle="--", linewidth=1)
    plt.text(len(subs) - 0.6, 0.265, "随机水平 25%", fontsize=9, color="#666")
    plt.xticks(x, subs)
    plt.ylim(0, 1.0)
    plt.ylabel("合并五折交叉验证准确率")
    plt.title("逐受试者准确率（两会话合并，五折交叉验证）")
    plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out / "per_subject_acc.png", dpi=150)
    plt.close()

    # 2) 混淆矩阵（融合模型，主协议聚合）
    npz = data["vote"][2]
    cm = confusion_matrix(npz["pooled_y"], npz["pooled_pred"], normalize="true")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CLASS_NAMES)
    ax.set_yticks(range(4), CLASS_NAMES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=10)
    acc = (npz["pooled_pred"] == npz["pooled_y"]).mean()
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"三模型投票融合的混淆矩阵（行归一化），总体准确率 {acc:.3f}")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    # 3) 汇总表
    lines = ["方法,协议,准确率均值,kappa均值"]
    for m, label in METHODS:
        accs = np.array([data[m][0][s] for s in subs])
        kaps = np.array([data[m][1][s] for s in subs])
        lines.append(f"{label},合并五折,{accs.mean():.4f},{kaps.mean():.4f}")
        lines.append(f"{label},跨会话,{data[m][3]:.4f},")
    (out / "summary.csv").write_text("\n".join(lines), encoding="utf-8-sig")
    print("\n".join(lines))

    # 4) 逐 trial 明细（融合模型，主协议）
    with open(out / "vote_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["受试者", "trial 序号", "真实类别", "预测类别", "是否正确"])
        for i, (s, y, p) in enumerate(zip(npz["pooled_sub"], npz["pooled_y"], npz["pooled_pred"])):
            w.writerow([s, i, CLASS_NAMES[int(y)], CLASS_NAMES[int(p)],
                        "正确" if y == p else "错误"])
    print(f"图与明细已保存到 {out}")


if __name__ == "__main__":
    main()
