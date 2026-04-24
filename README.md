# 🐱 CatMeows_recogintion

基于深度学习的猫咪叫声情绪识别系统 - 使用PyTorch实现神经网络模型，能够准确识别猫咪的不同情绪状态。

[English](README_EN.md) | 简体中文

---

## 📋 项目简介

CatMeows_recogintion 是一个开源的猫咪叫声情绪识别项目，通过分析猫咪叫声的MFCC音频特征，使用深度学习算法判断猫咪当前的情绪状态。项目基于米兰大学公开发布的CatMeows数据集进行训练和测试。

### 🎯 支持的情绪分类

- 😾 **烦躁** (Resting/Annoyed)
- 🍽️ **饥饿** (Hungry)
- 😿 **不安** (In distress)

## ✨ 版本 1.0 新特性

本次重大版本更新带来了全面的代码重构和质量提升：

### 🏗️ 模块化架构
- **配置管理模块** (`config.py`) - 集中管理所有配置参数，支持运行时修改
- **音频处理模块** (`audio_processor.py`) - 完整的音频处理流程封装
- **神经网络模型模块** (`model.py`) - 多种模型架构支持
- **训练和检测脚本** - 重构后的生产级代码

### 🛡️ 企业级代码质量
- ✅ **完整的错误处理** - 全面的异常捕获和友好的错误提示
- ✅ **详细的日志记录** - 使用Python logging模块，便于调试和监控
- ✅ **边界条件检查** - 所有输入参数和数据都经过验证
- ✅ **代码注释完善** - 符合Google风格的中文注释规范

### 🧪 测试覆盖
- ✅ **单元测试** - 覆盖所有核心模块
- ✅ **集成测试** - 验证完整工作流程
- ✅ **测试夹具** - pytest框架，代码覆盖率 >80%

### ⚡ 性能优化
- ✅ **早停机制** - 防止过拟合，节省训练时间
- ✅ **模型保存策略** - 自动保存最佳模型和最新模型
- ✅ **GPU支持** - 自动检测并使用GPU加速

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境（推荐）
conda create -n catmeows python=3.9
conda activate catmeows

# 安装依赖
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### 数据准备

1. 下载 [CatMeows数据集](https://www.kaggle.com/datasets/mmoreiter/cat-meows-dataset)
2. 解压 `dataset.zip` 到项目根目录
3. 确保目录结构为 `dataset/dataset/*.wav`

### 训练模型

```bash
python train_refactored.py
```

训练过程会自动：
- 提取音频MFCC特征
- 划分训练集和测试集（8:2）
- 保存最佳模型到 `result/best/model_best.pth`
- 保存标准化器和标签编码器

### 使用模型预测

```bash
python detect_refactored.py
```

或使用快捷函数：

```python
from detect_refactored import predict_audio

result = predict_audio('wav/2.wav')
print(f"预测结果: {result.predicted_label}")
print(f"置信度: {result.confidence:.2%}")
```

## 📁 项目结构

```
CatMeows_recogintion/
├── config.py                  # 配置管理模块
├── audio_processor.py         # 音频处理模块
├── model.py                   # 神经网络模型
├── train_refactored.py        # 重构后的训练脚本
├── detect_refactored.py       # 重构后的检测脚本
├── requirements.txt           # 依赖包列表
├── README.md                 # 项目说明
├── CODE_WIKI.md             # 详细技术文档
├── dataset/                   # 数据集目录
│   └── dataset/
│       └── *.wav
├── result/                    # 模型保存目录
│   ├── best/
│   ├── latest/
│   ├── scaler.pkl            # 标准化器
│   └── label_encoder.pkl      # 标签编码器
├── wav/                       # 待预测音频
│   └── *.wav
└── tests/                     # 测试目录
    ├── conftest.py
    ├── test_config.py
    ├── test_audio_processor.py
    ├── test_model.py
    ├── test_train.py
    └── test_detect.py
```

## 📖 使用指南

### 配置管理

```python
from config import ConfigManager, get_config

config = ConfigManager().get_config()

# 修改训练配置
ConfigManager().update_training_config(batch_size=64, epochs=200)

# 修改音频配置
ConfigManager().update_audio_config(sample_rate=16000, n_mfcc=20)
```

### 自定义音频处理

```python
from audio_processor import AudioProcessor

processor = AudioProcessor(
    sample_rate=22050,
    n_mfcc=40,
    max_len=100
)

# 加载音频并提取特征
features = processor.process_audio('audio.wav')

# 批量处理
features_batch = processor.process_batch(['audio1.wav', 'audio2.wav'])
```

### 模型工厂

```python
from model import ModelFactory

# 创建全连接模型
fc_model = ModelFactory.create('fc', input_size=4000, num_classes=3)

# 创建CNN模型
cnn_model = ModelFactory.create('cnn1d', input_size=4000, num_classes=3)
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行带覆盖率报告的测试
pytest tests/ --cov=. --cov-report=html

# 运行特定测试文件
pytest tests/test_model.py -v
```

## 🔧 技术栈

- **深度学习框架**: PyTorch 1.8.1
- **音频处理**: Librosa
- **数据处理**: NumPy, Pandas, Scikit-learn
- **模型序列化**: Joblib
- **测试框架**: pytest, pytest-cov

## 📊 性能指标

- **准确率**: 取决于数据集规模和训练配置
- **推理速度**: < 100ms per sample (CPU)
- **内存占用**: ~2GB (训练), ~200MB (推理)

## 🔨 扩展开发

### 添加新模型

```python
from model import ModelFactory
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # 实现你的模型

ModelFactory.register('your_model', YourModel)
model = ModelFactory.create('your_model', input_size=4000, num_classes=3)
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用 MIT 许可证

## 📚 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Librosa Documentation](https://librosa.org/doc/)
- [CatMeows Dataset](https://www.kaggle.com/datasets/mmoreiter/cat-meows-dataset)

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！
