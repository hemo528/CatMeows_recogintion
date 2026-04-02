"""
配置模块测试
测试配置管理和验证功能
"""

import pytest
from pathlib import Path
from dataclasses import dataclass

from config import (
    ConfigManager,
    AppConfig,
    ModelConfig,
    TrainingConfig,
    AudioConfig,
    PathConfig,
    get_config,
    get_device
)


class TestModelConfig:
    """模型配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = ModelConfig()

        assert config.input_size == 4000
        assert config.hidden_sizes == [512, 256, 64]
        assert config.dropout_rate == 0.0
        assert config.num_classes is None

    def test_custom_values(self):
        """测试自定义值"""
        config = ModelConfig(
            input_size=1000,
            num_classes=5,
            hidden_sizes=[256, 128],
            dropout_rate=0.3
        )

        assert config.input_size == 1000
        assert config.num_classes == 5
        assert config.hidden_sizes == [256, 128]
        assert config.dropout_rate == 0.3


class TestTrainingConfig:
    """训练配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = TrainingConfig()

        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.learning_rate == 0.001
        assert config.weight_decay == 0.0
        assert config.test_size == 0.2
        assert config.random_seed == 42
        assert config.save_interval == 10
        assert config.early_stopping_patience == 20
        assert config.use_cuda_if_available is True

    def test_custom_values(self):
        """测试自定义值"""
        config = TrainingConfig(
            batch_size=64,
            epochs=200,
            learning_rate=0.0001,
            early_stopping_patience=30
        )

        assert config.batch_size == 64
        assert config.epochs == 200
        assert config.learning_rate == 0.0001
        assert config.early_stopping_patience == 30


class TestAudioConfig:
    """音频配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = AudioConfig()

        assert config.sample_rate == 22050
        assert config.n_mfcc == 40
        assert config.max_len == 100
        assert config.hop_length == 512
        assert config.n_fft == 2048

    def test_custom_values(self):
        """测试自定义值"""
        config = AudioConfig(
            sample_rate=16000,
            n_mfcc=20,
            max_len=50
        )

        assert config.sample_rate == 16000
        assert config.n_mfcc == 20
        assert config.max_len == 50


class TestPathConfig:
    """路径配置测试"""

    def test_default_paths(self):
        """测试默认路径"""
        config = PathConfig()

        assert config.project_root == Path(__file__).parent.parent
        assert 'dataset' in str(config.dataset_dir)
        assert 'result' in str(config.result_dir)

    def test_custom_paths(self):
        """测试自定义路径"""
        custom_root = Path('/custom/path')
        config = PathConfig(project_root=custom_root)

        assert config.project_root == custom_root


class TestConfigManager:
    """配置管理器测试"""

    def test_singleton_pattern(self):
        """测试单例模式"""
        ConfigManager.reset()
        manager1 = ConfigManager()
        manager2 = ConfigManager()

        assert manager1 is manager2

    def test_get_config(self):
        """测试获取配置"""
        ConfigManager.reset()
        manager = ConfigManager()
        config = manager.get_config()

        assert isinstance(config, AppConfig)

    def test_update_training_config(self):
        """测试更新训练配置"""
        ConfigManager.reset()
        manager = ConfigManager()

        manager.update_training_config(batch_size=64, epochs=200)

        config = manager.get_config()
        assert config.training.batch_size == 64
        assert config.training.epochs == 200

    def test_update_audio_config(self):
        """测试更新音频配置"""
        ConfigManager.reset()
        manager = ConfigManager()

        manager.update_audio_config(sample_rate=16000, n_mfcc=20)

        config = manager.get_config()
        assert config.audio.sample_rate == 16000
        assert config.audio.n_mfcc == 20

    def test_update_model_config(self):
        """测试更新模型配置"""
        ConfigManager.reset()
        manager = ConfigManager()

        manager.update_model_config(dropout_rate=0.5)

        config = manager.get_config()
        assert config.model.dropout_rate == 0.5

    def test_validate_paths_nonexistent(self):
        """测试验证不存在的路径"""
        ConfigManager.reset()
        manager = ConfigManager()
        manager.update_config(
            paths=PathConfig(
                dataset_dir=Path('/nonexistent/dataset'),
                project_root=Path('/tmp')
            )
        )

        result = manager.validate_paths()

        assert result is False

    def test_get_device_cpu(self):
        """测试获取CPU设备"""
        ConfigManager.reset()
        manager = ConfigManager()
        manager.update_training_config(use_cuda_if_available=False)

        device = manager.get_device()

        assert device == 'cpu'

    def test_reset(self):
        """测试重置配置"""
        manager1 = ConfigManager()
        manager1.update_training_config(batch_size=128)

        ConfigManager.reset()

        manager2 = ConfigManager()
        config = manager2.get_config()

        assert config.training.batch_size == 32


class TestGetConfigFunction:
    """快捷函数测试"""

    def test_get_config_function(self):
        """测试获取配置快捷函数"""
        ConfigManager.reset()

        config = get_config()

        assert isinstance(config, AppConfig)

    def test_get_device_function(self):
        """测试获取设备快捷函数"""
        ConfigManager.reset()

        device = get_device()

        assert device in ['cpu', 'cuda']


class TestAppConfig:
    """应用配置测试"""

    def test_default_emotion_labels(self):
        """测试默认情绪标签"""
        config = AppConfig()

        assert config.emotion_labels[0] == '烦躁'
        assert config.emotion_labels[1] == '饥饿'
        assert config.emotion_labels[2] == '不安'

    def test_custom_emotion_labels(self):
        """测试自定义情绪标签"""
        custom_labels = {0: 'happy', 1: 'sad', 2: 'neutral'}
        config = AppConfig(emotion_labels=custom_labels)

        assert config.emotion_labels == custom_labels

    def test_nested_configs(self):
        """测试嵌套配置"""
        config = AppConfig()

        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.paths, PathConfig)


class TestConfigValidation:
    """配置验证测试"""

    def test_invalid_batch_size(self):
        """测试无效的batch_size"""
        ConfigManager.reset()
        manager = ConfigManager()

        manager.update_training_config(batch_size=-1)

        config = manager.get_config()
        assert config.training.batch_size == -1

    def test_invalid_learning_rate(self):
        """测试无效的学习率"""
        ConfigManager.reset()
        manager = ConfigManager()

        manager.update_training_config(learning_rate=0.0)

        config = manager.get_config()
        assert config.training.learning_rate == 0.0

    def test_empty_hidden_sizes(self):
        """测试空的隐藏层配置"""
        ConfigManager.reset()
        manager = ConfigManager()

        manager.update_model_config(hidden_sizes=[])

        config = manager.get_config()
        assert config.model.hidden_sizes == []
