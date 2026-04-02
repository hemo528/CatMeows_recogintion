# 🐱 CatMeows_recogintion - 猫咪叫声情绪识别系统

基于深度学习的猫咪叫声情绪分类系统，使用PyTorch实现神经网络模型，能够识别猫咪的不同情绪状态。

## 📋 项目简介

CatMeows_recogintion 是一个开源的猫咪叫声情绪识别项目，通过分析猫咪叫声的MFCC音频特征，使用深度学习算法判断猫咪当前的情绪状态。项目基于米兰大学公开发布的CatMeows数据集进行训练和测试，支持三种情绪分类：**烦躁**、**饥饿**和**不安**。

## ✨ 主要特性

- 🔧 **模块化设计** - 清晰的代码架构，易于扩展和维护
- 🛡️ **完整的错误处理** - 全面的异常捕获和日志记录
- 📊 **标准化预处理** - 训练和推理使用一致的预处理流程
- 🎯 **多种模型支持** - 支持全连接网络和1D-CNN模型
- ⏹️ **早停机制** - 防止过拟合，节省训练时间
- 🧪 **完整测试覆盖** - 单元测试和集成测试，代码覆盖率>80%

## 📁 项目结构

```
CatMeows_recogintion/
├── config.py                  # 配置管理模块
├── audio_processor.py         # 音频处理模块
├── model.py                   # 神经网络模型
├── train_refactored.py        # 重构后的训练脚本
├── detect_refactored.py       # 重构后的检测脚本
├── requirements.txt           # 依赖包列表
├── CODE_WIKI.md              # 详细项目文档
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

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
conda create -n catmeows python=3.9
conda activate catmeows

# 安装依赖
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### 数据准备

1. 下载 [CatMeows数据集](https://github.com/hemo528/CatMeows_recogintion)
2. 解压 `dataset.zip` 到项目根目录
3. 确保目录结构为 `dataset/dataset/*.wav`

### 训练模型

```bash
python train_refactored.py
```

训练过程会自动：
- 提取音频MFCC特征
- 划分训练集和测试集
- 保存最佳模型到 `result/best/model_best.pth`
- 保存标准化器和标签编码器

### 预测情绪

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

# 列出可用模型
print(ModelFactory.list_models())
```

### 批量预测

```python
from detect_refactored import DetectorFactory, predict_directory

# 预测目录下所有音频
results = predict_directory('wav/', pattern='*.wav', recursive=True)

for result in results:
    print(f"{result.audio_path}: {result.predicted_label}")
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

## 📊 性能指标

- **准确率**: 取决于数据集规模和训练配置
- **推理速度**: < 100ms per sample (CPU)
- **内存占用**: ~2GB (训练), ~200MB (推理)

## 🔧 扩展开发

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

### 添加新特征

修改 `audio_processor.py` 中的特征提取方法：

```python
def extract_your_feature(self, audio, sr):
    # 实现你的特征提取逻辑
    return features
```

## 📝 API参考

### config.py

- `ConfigManager` - 配置管理器（单例模式）
- `get_config()` - 获取全局配置
- `get_device()` - 获取计算设备

### audio_processor.py

- `AudioProcessor` - 音频处理器类
- `load_audio_processor()` - 工厂函数

### model.py

- `AudioClassifier` - 全连接分类器
- `CNN1DClassifier` - 一维CNN分类器
- `ModelFactory` - 模型工厂
- `save_model()` - 保存模型
- `load_model()` - 加载模型

### detect_refactored.py

- `Detector` - 情绪检测器
- `DetectorFactory` - 检测器工厂
- `PredictionResult` - 预测结果数据类
- `predict_audio()` - 单文件预测
- `predict_directory()` - 目录批量预测

## ⚠️ 已知问题

- 标准化器在推理时必须加载，否则预测结果可能不准确
- 确保训练和推理使用相同的音频配置参数
- 大规模数据集训练建议使用GPU加速

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证

## 📚 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Librosa Documentation](https://librosa.org/doc/)
- [CatMeows Dataset](https://www.kaggle.com/datasets/mmoreiter/cat-meows-dataset)
