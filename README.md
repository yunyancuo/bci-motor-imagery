# 基于脑电的运动想象四分类（BCI Competition IV 2a）

课程实践仓库 · 选题：脑电数据的分析与处理

- **数据集**：BCI Competition IV Dataset 2a，9 名受试者 × 2 会话，22 导 EEG @250Hz，四类运动想象（左手/右手/双脚/舌头），每会话 288 个 trial
- **基线模型**：CSP + LDA（共同空间模式 + 线性判别分析，BCI 经典流水线）
- **改进模型**：EEGNet（端到端时空卷积网络，约 2 千参数，为小样本 EEG 设计）
- **评估协议**：每受试者 T 会话训练 / E 会话测试（竞赛标准），报告准确率与 Cohen's kappa

## 目录结构

```text
bci-motor-imagery/
├── src/
│   ├── prepare.py    # gdf+真标签 → 每受试者 npz（8-30Hz 带通，0.5-4.0s epoch）
│   ├── eda.py        # ERD/ERS 示意（C3/C4 对侧效应）+ 频谱图
│   ├── csp_lda.py    # 基线：CSP+LDA
│   ├── eegnet.py     # 改进：EEGNet（逐受试者训练，留 20% 验证选模型）
│   └── evaluate.py   # 汇总对比、混淆矩阵、逐 trial 明细
└── 实验报告.md
```

## 快速开始（数据与环境在服务器上）

```bash
pip install mne scikit-learn torch numpy matplotlib scipy -i https://pypi.tuna.tsinghua.edu.cn/simple

python src/prepare.py --src-dir data_src --out-dir data/bci2a
python src/eda.py
python src/csp_lda.py            # 基线
python src/eegnet.py             # 改进（GPU 约 5 分钟 / 9 人）
python src/evaluate.py           # 出图与汇总
```

## 参考水平（2a 数据集文献常见值）

| 指标 | CSP 系基线 | 深度模型（EEGNet 级） |
|---|---|---|
| 准确率（9 人均值） | 0.58 ~ 0.70 | 0.60 ~ 0.72 |
| kappa | 0.44 ~ 0.60 | 0.47 ~ 0.62 |
