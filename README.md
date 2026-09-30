# MNIST-NumPy-PyTorch-CNN
这是一个面向 Python 初学者、同时连接深度学习与数字信号处理的课程项目。

项目路线：

```text
MNIST 数据理解
→ NumPy 线性 Softmax
→ NumPy 两层 MLP
→ PyTorch MLP
→ PyTorch CNN
→ 消融实验
→ 错误分析
→ 个人手写数字预测
→ 二维频域/DSP 分析
```

## 1. 安装

建议使用 Python 3.11 和虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

检查环境：

```powershell
python -c "import numpy, torch, torchvision, matplotlib, sklearn; print('environment ok')"
pytest -q
```

## 2. 训练 PyTorch 模型

第一次运行会自动下载 MNIST 到 `data/`。

```powershell
python -m src.train --model linear --epochs 5 --output-dir outputs/linear
python -m src.train --model mlp --epochs 5 --output-dir outputs/mlp
python -m src.train --model cnn --epochs 5 --output-dir outputs/cnn
```

快速冒烟训练可以使用较小数据集：

```powershell
python -m src.train --model cnn --epochs 1 --max-train 2000 --output-dir outputs/smoke
```

每次训练会保存：

- `best.pt`：验证集表现最好的模型；
- `history.json`：训练历史；
- `metadata.json`：参数、时间、设备和最佳验证准确率；
- `training_curves.png`：损失与准确率曲线；
- `sample_grid.png`：训练样本示例。

## 2.1 NumPy 原理实现

先做梯度检查：

```powershell
python -m src.numpy_train --model mlp --gradient-check --epochs 1 --max-train 1000
```

训练 NumPy 线性分类器或两层 MLP：

```powershell
python -m src.numpy_train --model linear --epochs 5 --output-dir outputs/numpy_linear
python -m src.numpy_train --model mlp --epochs 5 --output-dir outputs/numpy_mlp
```

NumPy 版本显式实现 Softmax、交叉熵、ReLU、反向传播和 mini-batch SGD，用于理解数学原理；PyTorch 版本用于更稳定的工程训练。

## 3. 测试集评估

```powershell
python -m src.evaluate `
  --checkpoint outputs/cnn/best.pt `
  --model cnn `
  --output-dir outputs/evaluation
```

输出包括：

- 测试集准确率和损失；
- 每类 Precision、Recall、F1；
- 混淆矩阵；
- 错误分类样本。

## 4. 个人手写数字

准备一张黑字白底或白字黑底的图片，例如 `my_digit.png`：

```powershell
python -m src.predict my_digit.png --checkpoint outputs/cnn/best.pt --model cnn
```

预处理过程包括灰度化、颜色反转判断、裁剪、缩放到 20×20、放入 28×28 画布和 MNIST 标准化。

## 5. DSP 扩展

```powershell
python -m src.dsp_analysis --data-dir data --output-dir outputs/dsp
```

该实验会展示数字图像、二维傅里叶频谱、低通滤波结果和高通滤波结果。

数字图像可以看作二维离散信号，CNN 卷积可以从信号处理角度理解为可学习的二维滤波器。

## 6. 数学重点

线性分类器：

\[
s=Wx+b
\]

Softmax：

\[
p_j=\frac{e^{s_j}}{\sum_k e^{s_k}}
\]

交叉熵：

\[
L=-\log p_y
\]

CNN 输出尺寸：

\[
H_{out}=\left\lfloor\frac{H+2P-K}{S}\right\rfloor+1
\]



