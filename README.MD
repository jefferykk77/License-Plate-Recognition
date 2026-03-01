# 🚗 License Plate Character Recognition

基于 CNN 深度学习的车牌字符识别系统，支持中国车牌 65 类字符分类。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-5C3EE8.svg)](https://opencv.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda)

## ✨ 功能特性

- 🔤 **65 类字符识别** - 支持数字、字母及 31 个中国省份简称
- 🧠 **CNN 深度学习** - 基于卷积神经网络，准确率高
- ⚡ **GPU 加速** - 支持 CUDA，识别速度快
- 🖼️ **自动字符分割** - 基于图像投影自动分割车牌字符
- 🎯 **端到端识别** - 输入车牌图片，直接输出识别结果

## 📋 支持字符

| 类型 | 字符 |
|------|------|
| 数字 | 0-9 |
| 字母 | A-Z (不含 I、O) |
| 省份 | 京、津、沪、渝、冀、晋、蒙、辽、吉、黑、苏、浙、皖、闽、赣、鲁、豫、鄂、湘、粤、桂、琼、川、贵、云、藏、陕、甘、青、宁、新 |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 12.4 (可选，用于 GPU 加速)

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/jefferykk77/License-Plate-Recognition.git
cd License-Plate-Recognition

# 安装依赖
pip install -r requirements.txt
```

### 运行识别

```bash
# 修改 predict.py 中的图片路径
IMAGE_PATH = r"./your_image.png"

# 运行预测
python predict.py
```

## 📁 项目结构

```
License-Plate-Recognition/
├── model.py              # CNN 模型定义 (LeNet 架构)
├── train.py              # 训练脚本
├── predict.py            # 预测脚本
├── model.pth             # 预训练模型权重
├── requirements.txt      # 依赖列表
├── datasets/             # 数据集目录
├── Cache_5b2ed614cefeb2a7.png  # 测试样例
└── README.md
```

## 🧠 模型架构

```
Input (1x32x32)
    ↓
Conv2d(1, 6, 5) + BN + MaxPool + ReLU
    ↓
Conv2d(6, 16, 5) + BN + MaxPool + ReLU
    ↓
Flatten
    ↓
Linear(400, 84) + ReLU
    ↓
Linear(84, 65)  # 65 类输出
```
## 🐛 常见问题

Q: 如何识别自己的图片？
A: 修改 predict.py 中的 IMAGE_PATH 变量

Q: 没有 GPU 能用吗？
A: 可以！自动检测 CUDA，没有则使用 CPU

Q: 支持新能源车牌吗？
A: 支持！普通蓝牌和新能源绿牌都可以
## 📝 更新日志
``` 
### v1.0.0 (2025-02)
- 🎉 初始版本发布
- ✨ 支持 65 类字符识别
- 🧠 基于 CNN 深度学习
- ⚡ 支持 GPU 加速

## 📄 许可证

MIT License

---

Made with ❤️ by Jeffery

如果这个项目对你有帮助，请给个 ⭐ Star！
