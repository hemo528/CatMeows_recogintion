"""
重构后的检测脚本 - 模块化设计，支持批量预测和多种模型类型
提供完整的错误处理、日志记录和结果输出功能
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from config import ConfigManager, get_config
from audio_processor import AudioProcessor, load_audio_processor
from model import load_model, AudioClassifier, ModelFactory


@dataclass
class PredictionResult:
    """预测结果数据类"""
    audio_path: str
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]

    def __str__(self) -> str:
        return (
            f"文件: {Path(self.audio_path).name}\n"
            f"预测: {self.predicted_label} (类别 {self.predicted_class})\n"
            f"置信度: {self.confidence:.2%}"
        )


class Detector:
    """音频情绪检测器"""

    def __init__(
        self,
        model: nn.Module,
        audio_processor: AudioProcessor,
        label_encoder,
        emotion_labels: Optional[Dict[int, str]] = None,
        device: Optional[str] = None
    ):
        """
        初始化检测器

        Args:
            model: 训练好的模型
            audio_processor: 音频处理器
            label_encoder: 标签编码器
            emotion_labels: 情绪标签映射
            device: 计算设备
        """
        self.model = model
        self.audio_processor = audio_processor
        self.label_encoder = label_encoder
        self.emotion_labels = emotion_labels or {0: '烦躁', 1: '饥饿', 2: '不安'}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"检测器初始化完成，使用设备: {self.device}")

    def predict_single(
        self,
        audio_path: Union[str, Path]
    ) -> PredictionResult:
        """
        对单个音频文件进行预测

        Args:
            audio_path: 音频文件路径

        Returns:
            预测结果

        Raises:
            FileNotFoundError: 音频文件不存在
            RuntimeError: 预测失败
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")

            features = self.audio_processor.process_audio(audio_path)
            features = features.reshape(1, -1)
            features = self.audio_processor.transform_features(features)

            inputs = torch.FloatTensor(features).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()

            confidence = probabilities[predicted_class].item()
            class_label = self.emotion_labels.get(
                predicted_class,
                self.label_encoder.inverse_transform([predicted_class])[0]
            )

            prob_dict = {
                self.emotion_labels.get(i, str(i)): prob.item()
                for i, prob in enumerate(probabilities)
            }

            result = PredictionResult(
                audio_path=str(audio_path),
                predicted_class=predicted_class,
                predicted_label=class_label,
                confidence=confidence,
                probabilities=prob_dict
            )

            self.logger.info(f"预测完成: {audio_path.name} -> {class_label}")
            return result

        except FileNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"预测失败 {audio_path}: {str(e)}")
            raise RuntimeError(f"预测失败: {str(e)}")

    def predict_batch(
        self,
        audio_paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> List[PredictionResult]:
        """
        批量预测多个音频文件

        Args:
            audio_paths: 音频文件路径列表
            show_progress: 是否显示进度

        Returns:
            预测结果列表
        """
        results = []
        total = len(audio_paths)

        self.logger.info(f"开始批量预测，共 {total} 个文件")

        for i, audio_path in enumerate(audio_paths, 1):
            if show_progress and i % 10 == 0:
                self.logger.info(f"进度: {i}/{total}")

            try:
                result = self.predict_single(audio_path)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"跳过文件 {audio_path}: {str(e)}")
                continue

        self.logger.info(f"批量预测完成，成功: {len(results)}/{total}")
        return results

    def predict_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.wav",
        recursive: bool = False,
        show_progress: bool = True
    ) -> List[PredictionResult]:
        """
        预测目录下所有匹配的音频文件

        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            recursive: 是否递归搜索子目录
            show_progress: 是否显示进度

        Returns:
            预测结果列表
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

        if recursive:
            audio_paths = list(directory.rglob(pattern))
        else:
            audio_paths = list(directory.glob(pattern))

        if not audio_paths:
            self.logger.warning(f"目录下没有找到匹配的文件: {directory}")
            return []

        self.logger.info(f"找到 {len(audio_paths)} 个音频文件")
        return self.predict_batch(audio_paths, show_progress)


class DetectorFactory:
    """检测器工厂类"""

    @staticmethod
    def from_config(
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        label_encoder_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> Detector:
        """
        从配置文件创建检测器

        Args:
            config_path: 配置文件路径
            model_path: 模型文件路径
            scaler_path: scaler文件路径
            label_encoder_path: 标签编码器文件路径
            device: 计算设备
            **kwargs: 其他配置参数

        Returns:
            配置好的Detector实例
        """
        config = ConfigManager().get_config()

        if model_path is None:
            model_path = config.paths.model_path

        if scaler_path is None:
            scaler_path = config.paths.scaler_path

        if label_encoder_path is None:
            label_encoder_path = config.paths.label_encoder_path

        audio_processor = load_audio_processor(
            scaler_path=scaler_path,
            sample_rate=config.audio.sample_rate,
            n_mfcc=config.audio.n_mfcc,
            max_len=config.audio.max_len
        )

        if not audio_processor.is_fitted:
            logging.warning("Scaler未加载，预测可能不准确")

        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint.get('model_config', {})
        model = AudioClassifier(
            input_size=model_config.get('input_size', 4000),
            num_classes=model_config.get('num_classes', 3),
            hidden_sizes=config.model.hidden_sizes,
            dropout_rate=0.0
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        try:
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
        except FileNotFoundError:
            logging.warning(f"标签编码器文件不存在: {label_encoder_path}")
            label_encoder = None

        return Detector(
            model=model,
            audio_processor=audio_processor,
            label_encoder=label_encoder,
            emotion_labels=config.emotion_labels,
            device=device
        )


def predict_audio(
    audio_path: Union[str, Path],
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    label_encoder_path: Optional[str] = None,
    device: Optional[str] = None
) -> PredictionResult:
    """
    快捷函数：对单个音频文件进行预测

    Args:
        audio_path: 音频文件路径
        model_path: 模型文件路径
        scaler_path: scaler文件路径
        label_encoder_path: 标签编码器文件路径
        device: 计算设备

    Returns:
        预测结果
    """
    detector = DetectorFactory.from_config(
        model_path=model_path,
        scaler_path=scaler_path,
        label_encoder_path=label_encoder_path,
        device=device
    )
    return detector.predict_single(audio_path)


def predict_directory(
    directory: Union[str, Path],
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    label_encoder_path: Optional[str] = None,
    pattern: str = "*.wav",
    recursive: bool = False,
    device: Optional[str] = None
) -> List[PredictionResult]:
    """
    快捷函数：预测目录下所有音频文件

    Args:
        directory: 目录路径
        model_path: 模型文件路径
        scaler_path: scaler文件路径
        label_encoder_path: 标签编码器文件路径
        pattern: 文件匹配模式
        recursive: 是否递归搜索
        device: 计算设备

    Returns:
        预测结果列表
    """
    detector = DetectorFactory.from_config(
        model_path=model_path,
        scaler_path=scaler_path,
        label_encoder_path=label_encoder_path,
        device=device
    )
    return detector.predict_directory(
        directory,
        pattern=pattern,
        recursive=recursive
    )


def main():
    """主检测函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = ConfigManager().get_config()

    try:
        detector = DetectorFactory.from_config()

        audio_path = input("请输入音频文件路径（或直接回车使用默认wav/2.wav）: ").strip()
        if not audio_path:
            audio_path = config.paths.wav_dir / '2.wav'
        else:
            audio_path = Path(audio_path)

        if audio_path.is_dir():
            results = detector.predict_directory(audio_path)

            print("\n" + "=" * 60)
            print("批量预测结果")
            print("=" * 60)
            for result in results:
                print(result)
                print("-" * 40)
        else:
            result = detector.predict_single(audio_path)

            print("\n" + "=" * 60)
            print("预测结果")
            print("=" * 60)
            print(result)

            print("\n各类别概率:")
            for label, prob in result.probabilities.items():
                print(f"  {label}: {prob:.2%}")

    except FileNotFoundError as e:
        logging.error(f"文件不存在: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"预测失败: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
