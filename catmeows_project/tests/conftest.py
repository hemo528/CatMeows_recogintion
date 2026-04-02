"""
pytest配置文件
定义测试夹具和配置
"""

import pytest
import sys
import os
from pathlib import Path
import tempfile
import shutil
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """创建临时目录用于测试"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_audio_data():
    """生成模拟音频数据（MFCC特征）"""
    np.random.seed(42)
    features = np.random.randn(40, 100).astype(np.float32)
    return features


@pytest.fixture
def sample_audio_batch():
    """生成模拟批量音频数据"""
    np.random.seed(42)
    batch_size = 8
    features = np.random.randn(batch_size, 4000).astype(np.float32)
    labels = np.random.randint(0, 3, size=batch_size)
    return features, labels


@pytest.fixture
def mock_audio_processor():
    """创建模拟音频处理器"""
    from audio_processor import AudioProcessor
    return AudioProcessor(
        sample_rate=22050,
        n_mfcc=40,
        max_len=100
    )


@pytest.fixture
def mock_model():
    """创建模拟模型"""
    from model import AudioClassifier
    return AudioClassifier(
        input_size=4000,
        num_classes=3,
        hidden_sizes=[512, 256, 64]
    )


@pytest.fixture
def trained_model(sample_audio_batch):
    """创建已训练的模型（用于集成测试）"""
    from model import AudioClassifier
    import torch.nn as nn
    import torch.optim as optim

    features, labels = sample_audio_batch

    model = AudioClassifier(
        input_size=4000,
        num_classes=3,
        hidden_sizes=[512, 256, 64]
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.LongTensor(labels)

    for _ in range(10):
        optimizer.zero_grad()
        outputs = model(features_tensor)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()

    return model


@pytest.fixture
def config_manager():
    """创建配置管理器"""
    from config import ConfigManager
    ConfigManager.reset()
    return ConfigManager()


@pytest.fixture(autouse=True)
def reset_config():
    """每个测试后重置配置"""
    yield
    from config import ConfigManager
    ConfigManager.reset()
