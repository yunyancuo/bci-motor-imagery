# 基于脑电信号的运动想象四分类（BCI Competition IV 2a）

课程实践仓库 · 选题：脑电数据的分析与处理

- **数据集**：BCI Competition IV Dataset 2a，9 名受试者 × 2 会话，22 导 EEG @250Hz，四类运动想象（左手/右手/双脚/舌头），每会话 288 个 trial
- **基线**：CSP + LDA（共同空间模式 + 线性判别分析）
- **改进**：FBCSP（6 子带滤波器组 CSP，竞赛冠军方法）、ShallowConvNet（端到端小网络，滑窗增强 + 多窗测试）
- **融合**：三模型逐 trial 多数投票
- **评估协议**：主协议 = 每受试者两会话合并、五折分层交叉验证；附加协议 = T 会话训练 → E 会话测试（跨天）

## 测试结果（9 名受试者）

| 方法 | 合并五折准确率 | kappa | 跨会话准确率 |
|---|---|---|---|
| CSP+LDA（基线） | 0.6715 ± 0.144 | 0.562 | 0.6339 |
| FBCSP（改进） | 0.7361 ± 0.123 | 0.648 | 0.6663 |
| ShallowConvNet | 0.6712 ± 0.145 | 0.562 | 0.5775 |
| **三模型投票（融合）** | **0.7492 ± 0.119** | **0.666** | **0.6871** |

跨会话 0.687（kappa 0.583）持平 BCI 竞赛冠军水平（kappa ≈ 0.57）；会话内 0.749 处于文献上沿。详见 `results/测试结果.md` 与 `实验报告.md`。

## 目录结构

```text
bci-motor-imagery/
├── src/
│   ├── prepare.py         # GDF+真标签 → npz（22 导、8-30Hz、0.5-4.0s epoch）
│   ├── eda.py             # ERD/ERS 对侧效应图 + 频谱图
│   ├── models.py          # ShallowConvNet
│   ├── run_experiment.py  # CSP / Shallow / 概率融合，双协议
│   ├── fbcsp.py           # 滤波器组 CSP
│   ├── vote_ensemble.py   # 三模型多数投票
│   └── evaluate.py        # 汇总表、逐受试者图、混淆矩阵、逐 trial 明细
├── results/               # 测试结果包
└── 实验报告.md
```

## 快速开始（数据与环境在服务器上）

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python src/prepare.py --src-dir data_src --out-dir data/bci2a
python src/eda.py
python src/run_experiment.py --model csp_lda --out-dir outputs/final/csp_lda
python src/run_experiment.py --model shallow --out-dir outputs/final/shallow   # GPU
python src/fbcsp.py          --out-dir outputs/final/fbcsp
python src/vote_ensemble.py
python src/evaluate.py
```
