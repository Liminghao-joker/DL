# Covid-19 Cases Prediction（DNN）

## 1. 背景与目标
- 任务：**回归**。用前 4 天的州级问卷与行为指标，预测第 5 天 `tested_positive` 百分比（目标列 `tested_positive.4`）。
- 评价指标：**MSE**。
- 目标：在 HW1 数据上完成端到端训练与提交，验证集 MSE ≈ **1.436**（当前实验）。

## 2. 快速开始
### 环境
- Python 3.10，PyTorch 2.7.1（CUDA 12.6）
- 安装：`pip install -r requirements.txt`

### 数据
- 从 Kaggle HW1 下载 `covid.train.csv` 与 `covid.test.csv`，放到 `./dataset/`。
- 列处理：删除 `id` 列；训练集最后一列为标签，其他为特征。

### 训练
- 直接运行：`python ClassCode.py`
- 日志：TensorBoard 写入 `runs/`，查看：`tensorboard --logdir runs`

### 评估/推理
- 训练脚本会按验证集 MSE 早停并保存最优权重到 `./models/`。
- 推理后自动生成 `pred.csv`，表头 `id,tested_positive`，`id` 为从 0 开始的行号。

## 3. 结果
| Metric | Value |
|--------|-------|
| Val MSE (best) | **1.436** |
| Train MSE (same step) | 1.135 |

- 训练曲线：见 TensorBoard（`Loss/Train`, `Loss/Validation`, `Loss/Difference`）。
- 提交文件：`pred.csv`。

## 4. 方法与实现要点
- 特征处理：连续特征做 z-score（仅用训练集统计）；one-hot 保持 0/1；删除 `id`。
- 模型：MLP `116 → 16 → 8 → 1`，ReLU，最后层无激活。
- 损失与优化：MSE；SGD（lr=1e-5, momentum=0.9）。可替换 Adam/AdamW 提升收敛。
- 正则化与稳定性：可选 BatchNorm、Dropout，梯度裁剪 1.0；早停基于验证 MSE。
- 可视化：TensorBoard 记录 Train/Val Loss 与差值。

## 5. 复现与随机性
- 固定随机种子；`cudnn.deterministic=True`，`benchmark=False`。
- 单卡/CPU 均可运行；批量大小默认 256，训练上限 300 epoch，8:2 划分 train/val。

## 6. 目录结构
- ClassCode.py
	- 读取数据、划分、训练、日志、保存与推理
- dataset
	`covid.train.csv` 和 `covid.test.csv`
- models
	- `model.ckpt`
- runs
	- Tensorboard`covid_experiment`
- pred.csv
- requirements.txt
- README.md

## 7. 已知问题与改进方向
- 优化器与学习率：当前 SGD+低 lr，建议改用 **Adam/AdamW**（**1e-3** 起试）并配合 ReduceLROnPlateau。
- 适度正则化：加入 Dropout/BatchNorm 或 weight decay，减小轻微过拟合。
- 模型容量：适当加宽/加深（如 116→256→128→64→1）并观察验证 MSE 变化。

## 8. 引用(Reference)
>Code
>>https://github.com/virginiakm1988/ML2022-Spring/blob/main/HW01/HW01.ipynb  
>
>Vedio
>>[2022-作业HW1_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Wv411h7kN?spm_id_from=333.788.videopod.episodes&vd_source=47dbec3f3db6a86044a31f482a95d4f0&p=11)
>
>Blog
>>[【李宏毅《机器学习》2022】作业1：COVID 19 Cases Prediction (Regression)_李宏毅2022作业-CSDN博客](https://blog.csdn.net/qq_41502322/article/details/123922649)
>>[ML起跑线 001：李宏毅老师机器学习（2022）HW1 示例代码学习笔记_李宏毅hw1-CSDN博客](https://blog.csdn.net/Lyndon_Ge/article/details/130570375)
>
>Kaggle
>>https://www.kaggle.com/competitions/ml2022spring-hw1
>>


