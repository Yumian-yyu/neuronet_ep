# BP 神经网络实验 —— 代码结构说明

实验要求只有三句话，所以 `experiments/` 里也只有三个文件，**一个实验一个文件，正文都在里面**。

| # | 实验要求原文 | 文件 |
|---|---|---|
| 1 | 观察比较 BP 网络**拓扑结构**及其它**各项参数**变化对于训练结果的影响 | `run_exp1_network.m` |
| 2 | 观察并分析**不同训练数据**对相同拓扑结构 BP 网络建模的影响 | `run_exp2_data.m` |
| 3 | 设计实现简单的 BP 网络，进行训练数据的**识别分类** | `run_exp3_classify.m` |

课堂第 6 节「感知器的验证」讲的是**怎么评价**（划分、K 折、混淆矩阵、四指标、ROC/AUC），
它是实验三的检验手段，不是独立实验，放在 `evaluate/` 里被调用。

## 一、目录结构

```
bp_code/
├── main.m                    一键运行三个实验（约 80 秒）
├── project_root.m            统一返回项目根目录
├── README.md
│
├── experiments/              3 个实验，正文自包含
│   ├── run_exp1_network.m      433 行：1.1 拓扑结构 + 1.2 各项参数
│   ├── run_exp2_data.m         270 行：6 种训练数据因素
│   └── run_exp3_classify.m     328 行：分类 + 划分 + K折 + ROC + 工具箱对照
│
├── tools/                    2 个，都是被多处共用的
│   ├── train_eval.m            43 行，统一"初始化→训练→评估"协议（5 处共用）
│   └── fig_utils.m             42 行，中文字体设置 + 导出图片（12 处共用）
│
├── evaluate/                 4 个，课堂检验方法
│   ├── eval_classifier.m       混淆矩阵 + 准确率 + P/R/F1 + AUC，一次给齐
│   ├── eval_cv.m               划分比例 + K 折交叉验证
│   ├── eval_roc.m              ROC 曲线
│   └── roc_auc.m               二分类 ROC/AUC 底层计算
│
├── core/                     8 个，手写 BP 的全部核心
│   ├── bp_init.m               权值初始化
│   ├── bp_forward.m            前向传播
│   ├── bp_backward.m           误差反传
│   ├── bp_train.m              训练主循环
│   ├── bp_loss.m / bp_predict.m
│   └── bp_act.m / bp_dact.m    激活函数及其导数
│
└── data/                     4 个
    ├── load_iris.m             读三个 txt
    ├── onehot.m                标签转 one-hot
    ├── norm_fit.m              只用训练集统计 μ、σ
    └── norm_apply.m            用 μ、σ 做变换
```

项目根目录下另有 `tools/`，放的是**出报告用**的工具，不参与实验运行：

```
神经网路实验/tools/
├── build_report.py           按模板生成实验报告 docx
└── draw_schematics.m         画网络结构图与算法流程图
```

## 二、运行

```matlab
cd D:\matlab_projects\神经网路实验\bp_code
main                      % 三个实验全跑，约 80 秒
main(struct('R',3))       % 快速版，重复次数少
```

## 三、三个实验各做什么

| 实验 | 内容 | 结果文件 |
|---|---|---|
| **一** | 1.1 拓扑：16 种结构 × 2 种输入条件（标准化/原始）<br>1.2 参数：学习率、动量、激活函数、初始化、训练算法、输出层、稳定性 | `exp1_topology.xlsx`<br>`exp1_params.xlsx` |
| **二** | 样本量、归一化方式、数据噪声、类别均衡性、样本顺序、特征子集 | `exp2_data.xlsx` |
| **三** | 3.1 分类（混淆矩阵 + 四指标）<br>3.2 划分比例 + 5 折交叉验证<br>3.3 ROC/AUC<br>3.4 与工具箱对照 | `exp3_class_*.xlsx`<br>`exp3_cv*.xlsx`<br>`exp3_roc_auc.xlsx`<br>`exp3_toolbox.xlsx` |

## 四、实验三与课堂检验方法的对应

| 课堂第 6 节 | 在实验三里的位置 |
|---|---|
| ① 训练/测试划分 70~80 / 20~30 | 3.2 划分比例对比 |
| ② K 折交叉验证 | 3.2 5 折交叉验证 |
| ③ 混淆矩阵 | 3.1 |
| ④ 准确率/精确率/召回率/F1 | 3.1 |
| ⑤ ROC 与 AUC | 3.3 |
| 实现正确性 | 3.4 与工具箱对照 |

## 五、输出

- `../figures/` —— 全部 PNG（200 dpi）
- `../results/` —— 全部 Excel 结果表与 `run_log.txt`

## 六、核心代码导航

算法本体约 150 行，代码里用 ①~⑦ 标了一条完整流水线，每处只有一行标注：

| 标注 | 含义 | 位置 |
|---|---|---|
| ① | 净输入 `z = W·a + b` | `core/bp_forward.m:16` |
| ② | 激活 `a = f(z)` | `core/bp_forward.m:19`、`:21` |
| ③ | 输出层误差 `δ = (y−t)·f′(z)` | `core/bp_backward.m:31` |
| ④ | 隐层误差 `δ = (Wᵀδ)·f′(z)` | `core/bp_backward.m:45` |
| ⑤ | 梯度 `dW = δ·aᵀ/N` | `core/bp_backward.m:42` |
| ⑥ | 更新权值 `W ← W − η·dW` | `core/bp_train.m:90` |
| ⑦ | 判停 | `core/bp_train.m:131` |

**建议阅读顺序**：

1. `core/bp_forward.m`（24 行）—— ①② 前向
2. `core/bp_backward.m`（49 行）—— ③④⑤ 反传
3. `core/bp_train.m` 主循环 —— ⑥⑦ 更新与判停
4. `core/bp_init.m` —— 初始化

## 七、约定

- 随机种子显式指定，实验可复现；
- 归一化参数只能由训练集统计；
- 评价统一走 `eval_classifier`，指标口径一致才可比。
