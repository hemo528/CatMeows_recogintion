"""
音频处理模块 - 负责音频文件的加载、特征提取和预处理
提供统一的音频处理接口，支持MFCC等多种特征提取方法
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import joblib


class AudioProcessingError(Exception):
    """音频处理异常"""
    pass


class AudioProcessor:
    """音频处理器类"""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 40,
        max_len: int = 100,
        hop_length: int = 512,
        n_fft: int = 2048
    ):
        """
        初始化音频处理器

        Args:
            sample_rate: 采样率 (Hz)
            n_mfcc: MFCC系数数量
            max_len: 特征序列最大长度
            hop_length: 帧移长度
            n_fft: FFT窗口大小
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.hop_length = hop_length
        self.n_fft = n_fft
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted = False
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_audio(
        self,
        audio_path: Union[str, Path],
        sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        加载音频文件

        Args:
            audio_path: 音频文件路径
            sr: 目标采样率，None则保持原始采样率

        Returns:
            (音频数据, 采样率) 元组

        Raises:
            AudioProcessingError: 音频加载失败
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise AudioProcessingError(f"音频文件不存在: {audio_path}")

            self.logger.debug(f"加载音频文件: {audio_path}")
            audio, sr = librosa.load(
                audio_path,
                sr=sr if sr is not None else self.sample_rate
            )

            if len(audio) == 0:
                raise AudioProcessingError(f"音频文件为空: {audio_path}")

            return audio, sr

        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            raise AudioProcessingError(f"加载音频失败 {audio_path}: {str(e)}")

    def extract_mfcc(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        提取MFCC特征

        Args:
            audio: 音频数据
            sr: 采样率

        Returns:
            MFCC特征矩阵 (n_mfcc, time)
        """
        if sr is None:
            sr = self.sample_rate

        self.logger.debug(f"提取MFCC特征: 采样率={sr}, n_mfcc={self.n_mfcc}")

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )

        return mfccs

    def pad_or_truncate(
        self,
        features: np.ndarray,
        max_len: Optional[int] = None
    ) -> np.ndarray:
        """
        调整特征长度至统一尺寸

        Args:
            features: 特征矩阵
            max_len: 目标长度，None则使用实例的max_len

        Returns:
            调整后的特征矩阵
        """
        if max_len is None:
            max_len = self.max_len

        if features.shape[1] < max_len:
            pad_width = max_len - features.shape[1]
            padded = np.pad(
                features,
                pad_width=((0, 0), (0, pad_width)),
                mode='constant',
                constant_values=0
            )
            self.logger.debug(
                f"特征补零: 原始长度={features.shape[1]}, 目标长度={max_len}"
            )
            return padded
        elif features.shape[1] > max_len:
            truncated = features[:, :max_len]
            self.logger.debug(
                f"特征截断: 原始长度={features.shape[1]}, 目标长度={max_len}"
            )
            return truncated
        return features

    def process_audio(
        self,
        audio_path: Union[str, Path],
        normalize: bool = False
    ) -> np.ndarray:
        """
        完整的音频处理流程

        Args:
            audio_path: 音频文件路径
            normalize: 是否使用已拟合的scaler进行标准化

        Returns:
            处理后的特征向量
        """
        audio, sr = self.load_audio(audio_path)
        mfccs = self.extract_mfcc(audio, sr)
        mfccs = self.pad_or_truncate(mfccs)
        features = mfccs.flatten()

        if normalize and self._is_fitted:
            features = self._scaler.transform([features])[0]

        return features

    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        normalize: bool = False
    ) -> np.ndarray:
        """
        批量处理音频文件

        Args:
            audio_paths: 音频文件路径列表
            normalize: 是否使用已拟合的scaler进行标准化

        Returns:
            特征矩阵 (n_samples, n_features)
        """
        features_list = []
        for path in audio_paths:
            features = self.process_audio(path, normalize=False)
            features_list.append(features)

        features_matrix = np.array(features_list)

        if normalize and self._is_fitted:
            features_matrix = self._scaler.transform(features_matrix)

        return features_matrix

    def fit_scaler(self, features: np.ndarray):
        """
        拟合标准化器

        Args:
            features: 特征矩阵 (n_samples, n_features)
        """
        self._scaler = StandardScaler()
        self._scaler.fit(features)
        self._is_fitted = True
        self.logger.info("StandardScaler 拟合完成")

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        标准化特征

        Args:
            features: 特征矩阵

        Returns:
            标准化后的特征矩阵

        Raises:
            AudioProcessingError: scaler未拟合
        """
        if not self._is_fitted:
            raise AudioProcessingError(
                "Scaler未拟合，请先调用fit_scaler方法"
            )
        return self._scaler.transform(features)

    def save_scaler(self, save_path: Union[str, Path]):
        """
        保存scaler到文件

        Args:
            save_path: 保存路径
        """
        if not self._is_fitted:
            raise AudioProcessingError("Scaler未拟合，无法保存")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, save_path)
        self.logger.info(f"Scaler已保存到: {save_path}")

    def load_scaler(self, load_path: Union[str, Path]):
        """
        从文件加载scaler

        Args:
            load_path: 加载路径
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise AudioProcessingError(f"Scaler文件不存在: {load_path}")

        self._scaler = joblib.load(load_path)
        self._is_fitted = True
        self.logger.info(f"Scaler已加载: {load_path}")

    def extract_label_from_filename(self, filename: str, delimiter: str = '_') -> str:
        """
        从文件名中提取标签

        Args:
            filename: 文件名
            delimiter: 分隔符

        Returns:
            提取的标签
        """
        return filename.split(delimiter)[0]

    @property
    def feature_size(self) -> int:
        """获取特征维度"""
        return self.n_mfcc * self.max_len

    @property
    def is_fitted(self) -> bool:
        """检查scaler是否已拟合"""
        return self._is_fitted


def load_audio_processor(
    scaler_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 22050,
    n_mfcc: int = 40,
    max_len: int = 100
) -> AudioProcessor:
    """
    工厂函数：创建并配置音频处理器

    Args:
        scaler_path: scaler文件路径，None则创建未拟合的处理器
        sample_rate: 采样率
        n_mfcc: MFCC系数数量
        max_len: 最大序列长度

    Returns:
        配置好的AudioProcessor实例
    """
    processor = AudioProcessor(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        max_len=max_len
    )

    if scaler_path is not None:
        processor.load_scaler(scaler_path)

    return processor
