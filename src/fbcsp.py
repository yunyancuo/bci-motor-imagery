"""FBCSP（滤波器组 CSP + LDA）：2a 竞赛冠军方法，作为改进模型之一。

把 8-32 Hz 拆成 6 个 4Hz 子带，每个子带独立学 4 个 CSP 空间滤波器，
拼接 24 维 log 方差特征后用 LDA 分类——不同受试者的 ERD/ERS 频段
有个体差异，滤波器组比单一 8-30 Hz 带更稳。

用法：
    python src/fbcsp.py --out-dir outputs/final/fbcsp
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import mne

from run_experiment import stratified_folds

BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
FS = 250


class FBCSP(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        self.csps_ = []
        feats = []
        for lo, hi in BANDS:
            Xb = self._band(X, lo, hi)
            csp = mne.decoding.CSP(n_components=self.n_components, reg="ledoit_wolf",
                                   log=True, norm_trace=False)
            feats.append(csp.fit_transform(Xb, y))
            self.csps_.append(csp)
        return self

    def transform(self, X):
        feats = [csp.transform(self._band(X, lo, hi)) for csp, (lo, hi) in zip(self.csps_, BANDS)]
        return np.concatenate(feats, axis=1)

    @staticmethod
    def _band(X, lo, hi):
        sos = butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
        return sosfiltfilt(sos, X, axis=2).astype(np.float32)


def run_fbcsp(Xtr, ytr, Xte, yte):
    clf = Pipeline([("fbcsp", FBCSP()), ("lda", LinearDiscriminantAnalysis())])
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)
    pred = prob.argmax(1)
    return float((pred == yte).mean()), float(cohen_kappa_score(yte, pred)), pred, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/bci2a")
    ap.add_argument("--out-dir", default="outputs/final/fbcsp")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pooled_rows, cross_rows = [], []
    p_preds, p_ys, p_subs, c_preds, c_ys, c_subs = [], [], [], [], [], []
    for sub in range(1, 10):
        try:
            T = np.load(Path(args.data_dir) / f"A{sub:02d}T.npz")
            E = np.load(Path(args.data_dir) / f"A{sub:02d}E.npz")
        except FileNotFoundError:
            continue
        X = np.concatenate([T["X"], E["X"]])
        y = np.concatenate([T["y"], E["y"]])
        folds = stratified_folds(y, k=5)
        accs, ks = [], []
        for te_idx in folds:
            tr_idx = np.concatenate([f for f in folds if f is not te_idx])
            acc, kap, pred, _ = run_fbcsp(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
            accs.append(acc); ks.append(kap)
            p_preds.append(pred); p_ys.append(y[te_idx])
            p_subs.append([f"A{sub:02d}"] * len(te_idx))
        pooled_rows.append((f"A{sub:02d}", float(np.mean(accs)), float(np.mean(ks))))

        acc, kap, pred, _ = run_fbcsp(T["X"], T["y"], E["X"], E["y"])
        cross_rows.append((f"A{sub:02d}", acc, kap))
        c_preds.append(pred); c_ys.append(E["y"])
        c_subs.append([f"A{sub:02d}"] * len(E["y"]))
        print(f"A{sub:02d}: 合并五折 {np.mean(accs):.4f} | 跨会话 {acc:.4f}", flush=True)

    for name, rows in [("pooled_cv10", pooled_rows), ("cross_session", cross_rows)]:
        accs = np.array([r[1] for r in rows])
        ks = np.array([r[2] for r in rows])
        print("")
        print(f"=== fbcsp / {name}（{len(rows)} 人）===")
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
