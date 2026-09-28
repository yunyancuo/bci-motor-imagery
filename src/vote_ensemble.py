"""三模型多数投票融合（CSP+LDA / ShallowConvNet / FBCSP），主协议事后组合。"""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score


def load(name):
    npz = np.load(Path("outputs/final") / name / "predictions.npz")
    return npz["pooled_pred"], npz["pooled_y"], npz["pooled_sub"]


def main():
    out = Path("outputs/final/vote")
    out.mkdir(parents=True, exist_ok=True)
    csp_pred, y, sub = load("csp_lda")
    sh_pred, _, _ = load("shallow")
    fb_pred, _, _ = load("fbcsp")
    # 三个硬预测投票，三者各异时偏向 FBCSP（单独准确率最高）
    from collections import Counter
    pred = []
    for a, b, c in zip(csp_pred, sh_pred, fb_pred):
        cnt = Counter([a, b, c])
        top = cnt.most_common()
        if top[0][1] >= 2:
            pred.append(top[0][0])
        else:
            pred.append(c)  # 三者各异时取 FBCSP
    pred = np.array(pred)

    acc = float((pred == y).mean())
    kappa = float(cohen_kappa_score(y, pred))
    subs = sorted(set(sub))
    rows = []
    for s in subs:
        m = sub == s
        rows.append((s, float((pred[m] == y[m]).mean()),
                     float(cohen_kappa_score(y[m], pred[m]))))
        print(f"{s}: 投票 {rows[-1][1]:.4f}")
    print("")
    print(f"=== 投票融合 / 合并五折（{len(rows)} 人）===")
    print(f"准确率 {acc:.4f} | kappa {kappa:.4f}")
    with open(out / "pooled_cv10.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["受试者", "准确率", "kappa"])
        w.writerows(rows)
        w.writerow(["均值", f"{acc:.4f}", f"{kappa:.4f}"])
    np.savez(out / "predictions.npz", pooled_pred=pred, pooled_y=y, pooled_sub=sub)

    # 跨会话协议同样做三模型投票
    csp_n = np.load(Path("outputs/final/csp_lda/predictions.npz"))
    sh_n = np.load(Path("outputs/final/shallow/predictions.npz"))
    fb_n = np.load(Path("outputs/final/fbcsp/predictions.npz"))
    from collections import Counter as C2
    cpred = []
    for a, b, c in zip(csp_n["cross_pred"], sh_n["cross_pred"], fb_n["cross_pred"]):
        cnt = C2([a, b, c])
        top = cnt.most_common()
        cpred.append(top[0][0] if top[0][1] >= 2 else c)
    cpred = np.array(cpred)
    cy, csub = csp_n["cross_y"], csp_n["cross_sub"]
    crows = []
    for s in sorted(set(csub)):
        m = csub == s
        crows.append((s, float((cpred[m] == cy[m]).mean()), float(cohen_kappa_score(cy[m], cpred[m]))))
    cacc = float((cpred == cy).mean())
    ck = float(cohen_kappa_score(cy, cpred))
    with open(out / "cross_session.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["受试者", "准确率", "kappa"])
        w.writerows(crows)
        w.writerow(["均值", f"{cacc:.4f}", f"{ck:.4f}"])
    print(f"跨会话投票: acc {cacc:.4f} kappa {ck:.4f}")


if __name__ == "__main__":
    main()
