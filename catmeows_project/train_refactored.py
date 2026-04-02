"""
重构后的训练脚本 - 模块化设计，添加完整的错误处理和日志记录
支持早停、模型保存、训练可视化等功能
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import ConfigManager, get_config, get_device
from audio_processor import AudioProcessor
from model import AudioClassifier, save_model, ModelFactory


class AudioDataset(Dataset):
    """自定义音频数据集类"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        初始化数据集

        Args:
            features: 特征矩阵 (n_samples, n_features)
            labels: 标签数组 (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        初始化早停器

        Args:
            patience: 容忍次数
            min_delta: 最小改善阈值
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_accuracy: float, epoch: int) -> bool:
        """
        检查是否应该早停

        Args:
            val_accuracy: 验证准确率
            epoch: 当前轮次

        Returns:
            是否应该早停
        """
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class Trainer:
    """模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str,
        config: Any
    ):
        """
        初始化训练器

        Args:
            model: 神经网络模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            criterion: 损失函数
            optimizer: 优化器
            device: 计算设备
            config: 配置对象
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.best_accuracy = 0.0
        self.train_history: List[Dict[str, float]] = []
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience
        )

    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for features, labels in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def evaluate(self) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return avg_loss, accuracy

    def train(self) -> Dict[str, Any]:
        """
        执行完整训练流程

        Returns:
            训练历史和最终结果
        """
        self.logger.info("=" * 60)
        self.logger.info("开始训练")
        self.logger.info("=" * 60)
        self.logger.info(
            f"训练配置: epochs={self.config.training.epochs}, "
            f"batch_size={self.config.training.batch_size}, "
            f"lr={self.config.training.learning_rate}"
        )
        self.logger.info(
            f"模型参数数量: {self.model.get_num_trainable_parameters()}"
        )

        start_time = datetime.now()

        for epoch in range(1, self.config.training.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.evaluate()

            self.train_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

            if epoch % self.config.training.save_interval == 0 or epoch == 1:
                self.logger.info(
                    f"Epoch {epoch}/{self.config.training.epochs} | "
                    f"训练损失: {train_loss:.4f} | "
                    f"验证损失: {val_loss:.4f} | "
                    f"验证准确率: {val_accuracy:.2f}%"
                )

            if epoch % self.config.training.save_interval == 0:
                latest_path = self.config.paths.latest_model_dir / 'model_latest.pth'
                save_model(self.model, str(latest_path))

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                best_path = self.config.paths.best_model_dir / 'model_best.pth'
                save_model(self.model, str(best_path))
                self.logger.info(f"*** 新的最佳模型! 准确率: {val_accuracy:.2f}% ***")

            if self.early_stopping(val_accuracy, epoch):
                self.logger.info(
                    f"早停触发于 epoch {epoch} "
                    f"(最佳epoch: {self.early_stopping.best_epoch}, "
                    f"最佳准确率: {self.early_stopping.best_score:.2f}%)"
                )
                break

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        results = {
            'best_accuracy': self.best_accuracy,
            'total_epochs': len(self.train_history),
            'training_time': training_time,
            'best_epoch': self.early_stopping.best_epoch,
            'history': self.train_history
        }

        self.logger.info("=" * 60)
        self.logger.info("训练完成")
        self.logger.info(f"最佳准确率: {self.best_accuracy:.2f}%")
        self.logger.info(f"训练时间: {training_time:.2f}秒")
        self.logger.info("=" * 60)

        return results


def load_and_preprocess_data(
    dataset_dir: Path,
    audio_processor: AudioProcessor
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    加载并预处理数据

    Args:
        dataset_dir: 数据集目录
        audio_processor: 音频处理器

    Returns:
        (特征矩阵, 标签数组, 标签编码器) 元组
    """
    logger = logging.getLogger('data_preprocessing')

    if not dataset_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")

    files = [f for f in os.listdir(dataset_dir) if f.endswith('.wav')]

    if not files:
        raise ValueError(f"数据集中没有找到.wav文件: {dataset_dir}")

    logger.info(f"找到 {len(files)} 个音频文件")

    data = {'filename': [], 'label': [], 'features': []}

    for i, filename in enumerate(files):
        if (i + 1) % 100 == 0:
            logger.info(f"处理进度: {i + 1}/{len(files)}")

        try:
            label = audio_processor.extract_label_from_filename(filename)
            filepath = dataset_dir / filename
            features = audio_processor.process_audio(filepath)

            data['filename'].append(filename)
            data['label'].append(label)
            data['features'].append(features)

        except Exception as e:
            logger.warning(f"处理文件失败 {filename}: {str(e)}")
            continue

    if not data['features']:
        raise RuntimeError("没有成功处理的音频文件")

    df = pd.DataFrame(data)
    logger.info(f"成功处理 {len(df)} 个文件")
    logger.info(f"类别分布:\n{df['label'].value_counts()}")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'].values)
    features = np.array(df['features'].tolist())

    audio_processor.fit_scaler(features)

    return features, labels, label_encoder


def setup_directories(config: Any):
    """设置必要的目录结构"""
    dirs = [
        config.paths.best_model_dir,
        config.paths.latest_model_dir,
        config.paths.result_dir
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"目录已创建/确认: {dir_path}")


def save_processors(
    audio_processor: AudioProcessor,
    label_encoder: LabelEncoder,
    config: Any
):
    """保存音频处理器和标签编码器"""
    audio_processor.save_scaler(config.paths.scaler_path)

    with open(config.paths.label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    logging.info(f"标签编码器已保存: {config.paths.label_encoder_path}")


def main():
    """主训练函数"""
    config = ConfigManager().get_config()
    device = ConfigManager().get_device()

    setup_directories(config)

    audio_processor = AudioProcessor(
        sample_rate=config.audio.sample_rate,
        n_mfcc=config.audio.n_mfcc,
        max_len=config.audio.max_len
    )

    try:
        features, labels, label_encoder = load_and_preprocess_data(
            config.paths.dataset_dir,
            audio_processor
        )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"数据加载失败: {str(e)}")
        sys.exit(1)

    save_processors(audio_processor, label_encoder, config)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=config.training.test_size,
        random_state=config.training.random_seed,
        stratify=labels
    )

    logging.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    X_train_scaled = audio_processor.transform_features(X_train)
    X_test_scaled = audio_processor.transform_features(X_test)

    train_dataset = AudioDataset(X_train_scaled, y_train)
    test_dataset = AudioDataset(X_test_scaled, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )

    num_classes = len(np.unique(labels))
    input_size = audio_processor.feature_size

    model = AudioClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_sizes=config.model.hidden_sizes,
        dropout_rate=config.model.dropout_rate
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config
    )

    results = trainer.train()

    logging.info("所有文件保存完成")
    return results


if __name__ == '__main__':
    main()
