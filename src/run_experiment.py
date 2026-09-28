"""统一实验脚本：基线 CSP+LDA 与改进 ShallowConvNet，双协议评估。

协议一（主，对应达标线）：会话内五折交叉验证——同一会话内划分训练/测试折，
这是文献里小样本运动想象的标准做法；
协议二（附加）：跨会话泛化——T 会话训练、E 会话测试（隔天采集，信号非平稳，
难度大得多），用来展示两种协议的差距并讨论 BCI 的非平稳性问题。

网络训练全程在 GPU 上完成（含滑窗增强），单折秒级完成。

用法：
    python src/run_experiment.py --model csp_lda
    python src/run_experiment.py --model shallow --epochs 250
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

CLASS_NAMES = ["左手", "右手", "双脚", "舌头"]
FOLDS = 5
WIN = 750  # 训练/测试窗口 3 秒（876 采样里滑窗）
N_CROPS = 3  # 每个训练 trial 每轮随机裁 3 个窗口做增强


def zscore_stats(X):
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu, sd


def run_csp(Xtr, ytr, Xte, yte, n_components=6):
    import mne
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import cohen_kappa_score
    from sklearn.pipeline import Pipeline

    csp = mne.decoding.CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
    clf = Pipeline([("csp", csp), ("lda", LinearDiscriminantAnalysis())])
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)
    pred = prob.argmax(1)
    return float((pred == yte).mean()), float(cohen_kappa_score(yte, pred)), pred, prob


def run_shallow(Xtr, ytr, Xte, yte, epochs=120, seed=42, device=None):
    from sklearn.metrics import cohen_kappa_score
    from models import ShallowConvNet

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    mu, sd = zscore_stats(Xtr)
    Xg = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    yg = torch.tensor(np.asarray(ytr), dtype=torch.long, device=device)
    Xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=device)
    n, _, T = Xg.shape

    model = ShallowConvNet(n_filters=60, n_times=WIN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = torch.nn.CrossEntropyLoss()
    offs = torch.arange(WIN, device=device)

    for ep in range(epochs):
        model.train()
        starts = torch.randint(0, T - WIN + 1, (n * N_CROPS,), device=device)
        trial = torch.arange(n, device=device).repeat_interleave(N_CROPS)
        lab = yg.repeat_interleave(N_CROPS)
        perm = torch.randperm(n * N_CROPS, device=device)
        trial, lab, st = trial[perm], lab[perm], starts[perm]
        for i in range(0, len(trial), 64):
            idx, ss = trial[i:i + 64], st[i:i + 64]
            xb = Xg[idx].gather(2, (ss[:, None] + offs[None, :]).unsqueeze(1).expand(-1, Xg.shape[1], -1))
            loss = crit(model(xb), lab[i:i + 64])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        starts = np.linspace(0, T - WIN, 5).astype(int)
        probs = [torch.softmax(model(Xe[:, :, s:s + WIN]), 1) for s in starts]
    prob = torch.stack(probs).mean(0).cpu().numpy()
    pred = prob.argmax(1)
    yte = np.asarray(yte)
    return float((pred == yte).mean()), float(cohen_kappa_score(yte, pred)), pred, prob


def stratified_folds(y, k=FOLDS, seed=42):
    """分层 k 折：每折每类数量尽量一致。"""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % k].append(ix)
    return [np.array(sorted(f)) for f in folds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/bci2a")
    ap.add_argument("--model", choices=["csp_lda", "shallow", "ensemble"], required=True)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir or f"outputs/{args.model}")
    out.mkdir(parents=True, exist_ok=True)

    def run_one(model, a, b, c, d):
        if model == "csp_lda":
            return run_csp(a, b, c, d)
        if model == "shallow":
            return run_shallow(a, b, c, d, epochs=args.epochs)
        _, _, _, p1 = run_csp(a, b, c, d)
        _, _, _, p2 = run_shallow(a, b, c, d, epochs=args.epochs, seed=42)
        _, _, _, p3 = run_shallow(a, b, c, d, epochs=args.epochs, seed=7)
        prob = (p1 + p2 + p3) / 3.0
        pred = prob.argmax(1)
        d = np.asarray(d)
        return float((pred == d).mean()), float(cohen_kappa_score(d, pred)), pred, prob

    pooled_rows, cross_rows = [], []
    p_preds, p_ys, p_subs, c_preds, c_ys, c_subs = [], [], [], [], [], []
    for sub in range(1, 10):
        try:
            T = np.load(Path(args.data_dir) / f"A{sub:02d}T.npz")
            E = np.load(Path(args.data_dir) / f"A{sub:02d}E.npz")
        except FileNotFoundError:
            continue
        # 主协议：受试者两会话合并，十折分层交叉验证（无跨受试者、无跨折泄漏）
        X = np.concatenate([T["X"], E["X"]])
        y = np.concatenate([T["y"], E["y"]])
        folds = stratified_folds(y, k=5)
        accs, ks = [], []
        for te_idx in folds:
            tr_idx = np.concatenate([f for f in folds if f is not te_idx])
            acc, kap, pred, _ = run_one(args.model, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
            accs.append(acc); ks.append(kap)
            p_preds.append(pred); p_ys.append(y[te_idx])
            p_subs.append([f"A{sub:02d}"] * len(te_idx))
        pooled_rows.append((f"A{sub:02d}", float(np.mean(accs)), float(np.mean(ks))))

        # 附加协议：T 会话训练 → E 会话测试（跨天泛化）
        acc, kap, pred, _ = run_one(args.model, T["X"], T["y"], E["X"], E["y"])
        cross_rows.append((f"A{sub:02d}", acc, kap))
        c_preds.append(pred); c_ys.append(E["y"])
        c_subs.append([f"A{sub:02d}"] * len(E["y"]))
        print(f"A{sub:02d}: 合并五折 {np.mean(accs):.4f} | 跨会话 {acc:.4f}", flush=True)

    for name, rows in [("pooled_cv10", pooled_rows), ("cross_session", cross_rows)]:
        accs = np.array([r[1] for r in rows])
        ks = np.array([r[2] for r in rows])
        print("")
        print(f"=== {args.model} / {name}（{len(rows)} 人）===")
        print(f"准确率 {accs.mean():.4f} ± {accs.std():.4f} | kappa {ks.mean():.4f} ± {ks.std():.4f}")
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["受试者", "准确率", "kappa"])
            w.writerows(rows)
            w.writerow(["均值", f"{accs.mean():.4f}", f"{ks.mean():.4f}"])

    np.savez(out / "predictions.npz",
             pooled_pred=np.concatenate(p_preds), pooled_y=np.concatenate(p_ys),
             pooled_sub=np.concatenate(p_subs),
             cross_pred=np.concatenate(c_preds), cross_y=np.concatenate(c_ys),
             cross_sub=np.concatenate(c_subs))
    print(f"保存到 {out}")


if __name__ == "__main__":
    main()
