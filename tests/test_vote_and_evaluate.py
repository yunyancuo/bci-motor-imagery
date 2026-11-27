"""多数投票融合与评估报表/图表生成模块单元测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.evaluate import generate_evaluation
from src.vote_ensemble import majority_vote, run_vote_ensemble


def test_majority_vote_rules_and_tiebreak() -> None:
    """验证多数投票规则：一致、两票多数、三者各异偏向决胜模型（FBCSP）。"""
    a = np.array([0, 1, 2, 0, 3])
    b = np.array([0, 1, 0, 2, 1])
    c = np.array([0, 2, 2, 2, 0])  # tie-breaker

    # idx 0: (0,0,0)->0; idx 1: (1,1,2)->1; idx 2: (2,0,2)->2;
    # idx 3: (0,2,2)->2; idx 4: (3,1,0)->各异取 c=0
    fused = majority_vote(a, b, c)
    assert np.array_equal(fused, np.array([0, 1, 2, 2, 0]))

    with pytest.raises(ValueError, match="长度不一致"):
        majority_vote([0, 1], [0], [0, 1])
    with pytest.raises(ValueError, match="不能为空"):
        majority_vote([], [], [])


def _write_mock_model_outputs(
    base_dir: Path,
    model_name: str,
    pooled_y: np.ndarray,
    pooled_pred: np.ndarray,
    pooled_sub: np.ndarray,
    cross_y: np.ndarray,
    cross_pred: np.ndarray,
    cross_sub: np.ndarray,
) -> None:
    """在临时目录写入单模型的 CSV 与 NPZ 模拟输出。"""
    m_dir = base_dir / model_name
    m_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        m_dir / "predictions.npz",
        pooled_pred=pooled_pred,
        pooled_y=pooled_y,
        pooled_sub=pooled_sub,
        cross_pred=cross_pred,
        cross_y=cross_y,
        cross_sub=cross_sub,
    )
    subs = sorted(set(pooled_sub.tolist()))
    for fname, yt, yp, sub_arr in [
        ("pooled_cv10.csv", pooled_y, pooled_pred, pooled_sub),
        ("cross_session.csv", cross_y, cross_pred, cross_sub),
    ]:
        with open(m_dir / fname, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["受试者", "准确率", "kappa"])
            accs = []
            for s in subs:
                mask = sub_arr == s
                acc = float(np.mean(yt[mask] == yp[mask]))
                kap = (acc - 0.25) / 0.75
                accs.append(acc)
                writer.writerow([s, f"{acc:.4f}", f"{kap:.4f}"])
            writer.writerow(["均值", f"{np.mean(accs):.4f}", "0.5000"])


def test_vote_ensemble_and_evaluate_end_to_end(tmp_path: Path) -> None:
    """端到端验证投票融合、标签对齐校验、图表生成与 Matplotlib 零句柄泄漏。"""
    in_dir = tmp_path / "outputs"
    vote_dir = in_dir / "vote"
    res_dir = tmp_path / "results"

    pooled_sub = np.array(["A01"] * 8 + ["A02"] * 8)
    pooled_y = np.array([0, 1, 2, 3, 0, 1, 2, 3] * 2)
    cross_sub = np.array(["A01"] * 4 + ["A02"] * 4)
    cross_y = np.array([0, 1, 2, 3] * 2)

    # 构造三个有互补性的预测：融合准确率应严格优于最差单模型
    csp_p = np.array([0, 1, 2, 0, 0, 1, 1, 3, 0, 1, 2, 3, 1, 1, 2, 3])
    sh_p = np.array([0, 2, 2, 3, 0, 1, 2, 3, 0, 0, 2, 3, 0, 1, 2, 3])
    fb_p = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

    csp_c = np.array([0, 1, 0, 3, 0, 1, 2, 0])
    sh_c = np.array([0, 2, 2, 3, 0, 1, 2, 3])
    fb_c = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    _write_mock_model_outputs(
        in_dir, "csp_lda", pooled_y, csp_p, pooled_sub, cross_y, csp_c, cross_sub
    )
    _write_mock_model_outputs(
        in_dir, "shallow", pooled_y, sh_p, pooled_sub, cross_y, sh_c, cross_sub
    )
    _write_mock_model_outputs(
        in_dir, "fbcsp", pooled_y, fb_p, pooled_sub, cross_y, fb_c, cross_sub
    )

    metrics = run_vote_ensemble(in_dir, vote_dir)
    assert metrics["pooled_acc"] == pytest.approx(1.0)
    assert metrics["cross_acc"] == pytest.approx(1.0)
    assert (vote_dir / "pooled_cv10.csv").exists()
    assert (vote_dir / "cross_session.csv").exists()
    expected_keys = {
        "pooled_pred",
        "pooled_y",
        "pooled_sub",
        "cross_pred",
        "cross_y",
        "cross_sub",
    }
    with np.load(vote_dir / "predictions.npz") as npz:
        assert expected_keys <= set(npz.files)

    # 运行 evaluate 生成图表与汇总 CSV，并校验所有图形已关闭（防内存泄漏）
    plt.close("all")
    lines = generate_evaluation(in_dir, res_dir)
    assert len(plt.get_fignums()) == 0
    assert (res_dir / "per_subject_acc.png").stat().st_size > 1000
    assert (res_dir / "confusion_matrix.png").stat().st_size > 1000
    assert (res_dir / "summary.csv").exists()
    assert (res_dir / "vote_predictions.csv").exists()
    assert any("三模型投票（融合）,合并五折,1.0000,1.0000" in line for line in lines)

    with open(res_dir / "vote_predictions.csv", encoding="utf-8-sig") as f:
        detail_rows = list(csv.DictReader(f))
    assert len(detail_rows) == len(pooled_y)
    assert all(r["是否正确"] == "正确" for r in detail_rows)


def test_vote_ensemble_rejects_misaligned_labels(tmp_path: Path) -> None:
    """验证多模型标签或受试者顺序不对齐时投票脚本立即拦截并报错。"""
    in_dir = tmp_path / "misaligned"
    sub = np.array(["A01"] * 4)
    y_ok = np.array([0, 1, 2, 3])
    y_bad = np.array([3, 2, 1, 0])

    _write_mock_model_outputs(in_dir, "csp_lda", y_ok, y_ok, sub, y_ok, y_ok, sub)
    _write_mock_model_outputs(in_dir, "shallow", y_ok, y_ok, sub, y_ok, y_ok, sub)
    _write_mock_model_outputs(in_dir, "fbcsp", y_bad, y_bad, sub, y_ok, y_ok, sub)

    with pytest.raises(ValueError, match="真实标签 y 与基线不一致"):
        run_vote_ensemble(in_dir, in_dir / "vote")
