"""
训练模块测试
测试训练器类和训练流程
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from train_refactored import (
    AudioDataset,
    EarlyStopping,
    Trainer,
    load_and_preprocess_data,
    setup_directories,
    save_processors
)


class TestAudioDataset:
    """音频数据集测试"""

    def test_init(self):
        """测试初始化"""
        features = np.random.randn(100, 4000)
        labels = np.random.randint(0, 3, size=100)

        dataset = AudioDataset(features, labels)

        assert len(dataset) == 100
        assert isinstance(dataset.features, torch.Tensor)
        assert isinstance(dataset.labels, torch.Tensor)

    def test_len(self):
        """测试数据集长度"""
        features = np.random.randn(50, 4000)
        labels = np.random.randint(0, 3, size=50)

        dataset = AudioDataset(features, labels)

        assert len(dataset) == 50

    def test_getitem(self):
        """测试获取单个样本"""
        features = np.random.randn(100, 4000)
        labels = np.random.randint(0, 3, size=100)

        dataset = AudioDataset(features, labels)

        feat, label = dataset[0]

        assert isinstance(feat, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert feat.shape == (4000,)
        assert label.item() in [0, 1, 2]

    def test_getitem_random_index(self):
        """测试随机索引访问"""
        features = np.random.randn(100, 4000)
        labels = np.random.randint(0, 3, size=100)

        dataset = AudioDataset(features, labels)

        idx = 50
        feat, label = dataset[idx]

        assert feat.shape == (4000,)
        assert label.item() == labels[idx]


class TestEarlyStopping:
    """早停机制测试"""

    def test_init(self):
        """测试初始化"""
        es = EarlyStopping(patience=5, min_delta=0.01)

        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.counter == 0
        assert es.best_score is None
        assert es.early_stop is False

    def test_first_evaluation(self):
        """测试首次评估"""
        es = EarlyStopping(patience=5)

        should_stop = es(0.8, epoch=1)

        assert not should_stop
        assert es.best_score == 0.8
        assert es.best_epoch == 1
        assert es.counter == 0

    def test_improvement(self):
        """测试性能提升"""
        es = EarlyStopping(patience=5, min_delta=0.01)

        es(0.8, epoch=1)
        should_stop = es(0.85, epoch=2)

        assert not should_stop
        assert es.best_score == 0.85
        assert es.best_epoch == 2
        assert es.counter == 0

    def test_no_improvement(self):
        """测试性能未提升"""
        es = EarlyStopping(patience=5, min_delta=0.01)

        es(0.8, epoch=1)
        es(0.79, epoch=2)

        assert es.counter == 1

    def test_early_stop_triggered(self):
        """测试早停触发"""
        es = EarlyStopping(patience=3, min_delta=0.01)

        es(0.8, epoch=1)
        es(0.75, epoch=2)
        es(0.75, epoch=3)
        should_stop = es(0.75, epoch=4)

        assert should_stop
        assert es.early_stop is True

    def test_min_delta_threshold(self):
        """测试最小改善阈值"""
        es = EarlyStopping(patience=5, min_delta=0.1)

        es(0.8, epoch=1)
        should_stop = es(0.85, epoch=2)

        assert should_stop
        assert es.counter == 1


class TestTrainer:
    """训练器测试"""

    @pytest.fixture
    def trainer_components(self, mock_model):
        """创建训练器组件"""
        features = np.random.randn(80, 4000)
        labels = np.random.randint(0, 3, size=80)

        train_dataset = AudioDataset(features, labels)
        test_dataset = AudioDataset(features[:20], labels[:20])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)

        from config import ConfigManager
        ConfigManager.reset()
        config = ConfigManager().get_config()

        trainer = Trainer(
            model=mock_model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device='cpu',
            config=config
        )

        return trainer, mock_model

    def test_init(self, trainer_components):
        """测试初始化"""
        trainer, _ = trainer_components

        assert trainer.best_accuracy == 0.0
        assert isinstance(trainer.train_history, list)
        assert isinstance(trainer.early_stopping, EarlyStopping)

    def test_train_epoch(self, trainer_components):
        """测试单个epoch训练"""
        trainer, model = trainer_components

        initial_loss = None
        for features, labels in trainer.train_loader:
            output = model(features)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, labels)
            initial_loss = loss.item()
            break

        loss = trainer.train_epoch()

        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate(self, trainer_components):
        """测试模型评估"""
        trainer, _ = trainer_components

        loss, accuracy = trainer.evaluate()

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100

    def test_train_short(self, trainer_components):
        """测试短期训练"""
        trainer, _ = trainer_components

        trainer.config.training.epochs = 2
        trainer.config.training.save_interval = 2

        results = trainer.train()

        assert 'best_accuracy' in results
        assert 'total_epochs' in results
        assert 'training_time' in results
        assert 'history' in results
        assert results['total_epochs'] <= 2


class TestSetupDirectories:
    """目录设置测试"""

    def test_setup_directories(self, temp_dir):
        """测试目录创建"""
        from config import ConfigManager
        ConfigManager.reset()
        config = ConfigManager().get_config()

        config.paths.result_dir = temp_dir / 'result'
        config.paths.best_model_dir = temp_dir / 'result' / 'best'
        config.paths.latest_model_dir = temp_dir / 'result' / 'latest'

        setup_directories(config)

        assert config.paths.best_model_dir.exists()
        assert config.paths.latest_model_dir.exists()


class TestLoadAndPreprocessData:
    """数据加载测试"""

    def test_missing_dataset_directory(self, temp_dir):
        """测试数据集目录不存在"""
        from config import ConfigManager
        ConfigManager.reset()
        config = ConfigManager().get_config()

        config.paths.dataset_dir = temp_dir / 'nonexistent'

        from audio_processor import AudioProcessor
        processor = AudioProcessor()

        with pytest.raises(FileNotFoundError, match="数据集目录不存在"):
            load_and_preprocess_data(config.paths.dataset_dir, processor)


class TestSaveProcessors:
    """处理器保存测试"""

    def test_save_processors(self, temp_dir, mock_audio_processor):
        """测试保存处理器"""
        from config import ConfigManager
        ConfigManager.reset()
        config = ConfigManager().get_config()

        config.paths.result_dir = temp_dir / 'result'
        config.paths.scaler_path = temp_dir / 'result' / 'scaler.pkl'
        config.paths.label_encoder_path = temp_dir / 'result' / 'encoder.pkl'

        train_data = np.random.randn(100, 4000)
        mock_audio_processor.fit_scaler(train_data)

        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(['A', 'B', 'C'])

        save_processors(mock_audio_processor, label_encoder, config)

        assert config.paths.scaler_path.exists()
        assert config.paths.label_encoder_path.exists()


class TestTrainingIntegration:
    """训练集成测试"""

    def test_full_training_pipeline(self, temp_dir):
        """测试完整训练流程"""
        from config import ConfigManager
        ConfigManager.reset()
        config = ConfigManager().get_config()

        config.paths.result_dir = temp_dir / 'result'
        config.paths.best_model_dir = temp_dir / 'result' / 'best'
        config.paths.latest_model_dir = temp_dir / 'result' / 'latest'
        config.paths.scaler_path = temp_dir / 'result' / 'scaler.pkl'
        config.paths.label_encoder_path = temp_dir / 'result' / 'encoder.pkl'

        setup_directories(config)

        features = np.random.randn(100, 4000)
        labels = np.random.randint(0, 3, size=100)

        train_features = features[:80]
        train_labels = labels[:80]
        test_features = features[80:]
        test_labels = labels[80:]

        train_dataset = AudioDataset(train_features, train_labels)
        test_dataset = AudioDataset(test_features, test_labels)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False
        )

        from model import AudioClassifier
        model = AudioClassifier(input_size=4000, num_classes=3)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device='cpu',
            config=config
        )

        trainer.config.training.epochs = 5
        trainer.config.training.save_interval = 5

        results = trainer.train()

        assert results['total_epochs'] == 5
        assert len(results['history']) == 5
