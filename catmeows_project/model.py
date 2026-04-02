"""
神经网络模型模块 - 定义音频分类模型架构
支持多种模型类型，提供统一的模型接口
"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioClassifier(nn.Module):
    """音频分类全连接神经网络"""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout_rate: float = 0.0
    ):
        """
        初始化音频分类器

        Args:
            input_size: 输入特征维度
            num_classes: 分类类别数量
            hidden_sizes: 隐藏层神经元数量列表，默认[512, 256, 64]
            dropout_rate: Dropout比率
        """
        super(AudioClassifier, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 64]

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.logger = logging.getLogger(self.__class__.__name__)

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

        self._init_weights()
        self.logger.info(
            f"模型初始化: input_size={input_size}, "
            f"hidden_sizes={hidden_sizes}, num_classes={num_classes}"
        )

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, input_size)

        Returns:
            输出 logits (batch_size, num_classes)
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别概率

        Args:
            x: 输入张量

        Returns:
            各类别概率 (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别

        Args:
            x: 输入张量

        Returns:
            预测类别 (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def get_num_parameters(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNN1DClassifier(nn.Module):
    """一维卷积神经网络分类器"""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        n_mfcc: int = 40,
        dropout_rate: float = 0.3
    ):
        """
        初始化1D-CNN分类器

        Args:
            input_size: 输入特征维度 (n_mfcc * max_len)
            num_classes: 分类类别数量
            n_mfcc: MFCC系数维度
            dropout_rate: Dropout比率
        """
        super(CNN1DClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.n_mfcc = n_mfcc
        self.logger = logging.getLogger(self.__class__.__name__)

        max_len = input_size // n_mfcc

        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        self.logger.info(
            f"CNN1D模型初始化: input_size={input_size}, "
            f"num_classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, input_size)

        Returns:
            输出 logits (batch_size, num_classes)
        """
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_mfcc, -1)

        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别概率"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


class ModelFactory:
    """模型工厂类"""

    _models = {
        'fc': AudioClassifier,
        'cnn1d': CNN1DClassifier
    }

    @classmethod
    def create(
        cls,
        model_type: str,
        input_size: int,
        num_classes: int,
        **kwargs
    ) -> nn.Module:
        """
        创建模型实例

        Args:
            model_type: 模型类型 ('fc' 或 'cnn1d')
            input_size: 输入特征维度
            num_classes: 分类类别数量
            **kwargs: 其他模型参数

        Returns:
            模型实例
        """
        if model_type not in cls._models:
            raise ValueError(
                f"未知的模型类型: {model_type}, "
                f"可选: {list(cls._models.keys())}"
            )

        model_class = cls._models[model_type]
        return model_class(
            input_size=input_size,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def register(cls, name: str, model_class: type):
        """注册新模型类型"""
        cls._models[name] = model_class

    @classmethod
    def list_models(cls) -> List[str]:
        """列出所有可用的模型类型"""
        return list(cls._models.keys())


def save_model(
    model: nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None
):
    """
    保存模型及相关信息

    Args:
        model: 模型实例
        save_path: 保存路径
        optimizer: 优化器（可选）
        epoch: 当前训练轮次
        metrics: 训练指标
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_config': {
            'input_size': model.input_size,
            'num_classes': model.num_classes
        }
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, save_path)
    logging.info(f"模型已保存: {save_path}")


def load_model(
    load_path: str,
    model_type: str = 'fc',
    **model_kwargs
) -> nn.Module:
    """
    加载模型

    Args:
        load_path: 模型文件路径
        model_type: 模型类型
        **model_kwargs: 模型参数

    Returns:
        加载的模型实例
    """
    checkpoint = torch.load(load_path, map_location='cpu')

    config = checkpoint.get('model_config', {})
    input_size = model_kwargs.pop('input_size', config.get('input_size', 4000))
    num_classes = model_kwargs.pop('num_classes', config.get('num_classes', 3))

    model = ModelFactory.create(
        model_type,
        input_size=input_size,
        num_classes=num_classes,
        **model_kwargs
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"模型已加载: {load_path}")

    return model
