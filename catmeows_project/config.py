"""
配置管理模块 - 集中管理所有配置参数
提供默认配置和配置验证功能
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging


@dataclass
class ModelConfig:
    """模型配置"""
    input_size: int = 4000
    hidden_sizes: list = field(default_factory=lambda: [512, 256, 64])
    dropout_rate: float = 0.0
    num_classes: Optional[int] = None


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    test_size: float = 0.2
    random_seed: int = 42
    save_interval: int = 10
    early_stopping_patience: int = 20
    use_cuda_if_available: bool = True


@dataclass
class AudioConfig:
    """音频处理配置"""
    sample_rate: int = 22050
    n_mfcc: int = 40
    max_len: int = 100
    hop_length: int = 512
    n_fft: int = 2048


@dataclass
class PathConfig:
    """路径配置"""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    dataset_dir: Path = field(default_factory=lambda: Path('dataset/dataset'))
    result_dir: Path = field(default_factory=lambda: Path('result'))
    best_model_dir: Path = field(default_factory=lambda: Path('result/best'))
    latest_model_dir: Path = field(default_factory=lambda: Path('result/latest'))
    wav_dir: Path = field(default_factory=lambda: Path('wav'))
    model_path: Path = field(default_factory=lambda: Path('result/best/model_best.pth'))
    scaler_path: Path = field(default_factory=lambda: Path('result/scaler.pkl'))
    label_encoder_path: Path = field(default_factory=lambda: Path('result/label_encoder.pkl'))

    def __post_init__(self):
        self.dataset_dir = self.project_root / self.dataset_dir
        self.result_dir = self.project_root / self.result_dir
        self.best_model_dir = self.project_root / self.best_model_dir
        self.latest_model_dir = self.project_root / self.latest_model_dir
        self.wav_dir = self.project_root / self.wav_dir
        self.model_path = self.project_root / self.model_path
        self.scaler_path = self.project_root / self.scaler_path
        self.label_encoder_path = self.project_root / self.label_encoder_path


@dataclass
class AppConfig:
    """应用配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    emotion_labels: Dict[int, str] = field(default_factory=lambda: {
        0: '烦躁',
        1: '饥饿',
        2: '不安'
    })
    log_level: str = 'INFO'
    log_file: Optional[str] = 'training.log'


class ConfigManager:
    """配置管理器"""

    _instance = None
    _config: Optional[AppConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._config = self._create_default_config()
        self._setup_logging()

    @staticmethod
    def _create_default_config() -> AppConfig:
        """创建默认配置"""
        return AppConfig()

    def get_config(self) -> AppConfig:
        """获取配置对象"""
        return self._config

    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def update_training_config(self, **kwargs):
        """更新训练配置"""
        for key, value in kwargs.items():
            if hasattr(self._config.training, key):
                setattr(self._config.training, key, value)

    def update_audio_config(self, **kwargs):
        """更新音频配置"""
        for key, value in kwargs.items():
            if hasattr(self._config.audio, key):
                setattr(self._config.audio, key, value)

    def update_model_config(self, **kwargs):
        """更新模型配置"""
        for key, value in kwargs.items():
            if hasattr(self._config.model, key):
                setattr(self._config.model, key, value)

    def get_device(self) -> str:
        """获取计算设备"""
        import torch
        config = self._config
        if config.training.use_cuda_if_available and torch.cuda.is_available():
            device = 'cuda'
            logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logging.info("使用CPU进行计算")
        return device

    def _setup_logging(self):
        """设置日志记录"""
        log_level = getattr(logging, self._config.log_level.upper(), logging.INFO)
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        handlers = [logging.StreamHandler()]

        if self._config.log_file:
            log_path = self._config.paths.project_root / self._config.log_file
            handlers.append(logging.FileHandler(log_path, encoding='utf-8'))

        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt=date_format,
            handlers=handlers
        )

    def validate_paths(self) -> bool:
        """验证关键路径"""
        config = self._config
        errors = []

        if not config.paths.dataset_dir.exists():
            errors.append(f"数据集目录不存在: {config.paths.dataset_dir}")

        if errors:
            for error in errors:
                logging.error(error)
            return False
        return True

    @classmethod
    def reset(cls):
        """重置配置（用于测试）"""
        cls._config = cls._create_default_config()
        cls._instance = None


def get_config() -> AppConfig:
    """获取全局配置（快捷函数）"""
    return ConfigManager().get_config()


def get_device() -> str:
    """获取计算设备（快捷函数）"""
    return ConfigManager().get_device()
