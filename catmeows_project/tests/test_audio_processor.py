"""
音频处理模块测试
测试AudioProcessor类的各项功能
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from audio_processor import (
    AudioProcessor,
    AudioProcessingError,
    load_audio_processor
)


class TestAudioProcessor:
    """AudioProcessor类测试"""

    def test_init_default_params(self):
        """测试默认参数初始化"""
        processor = AudioProcessor()

        assert processor.sample_rate == 22050
        assert processor.n_mfcc == 40
        assert processor.max_len == 100
        assert processor.hop_length == 512
        assert processor.n_fft == 2048
        assert not processor.is_fitted

    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        processor = AudioProcessor(
            sample_rate=16000,
            n_mfcc=20,
            max_len=50
        )

        assert processor.sample_rate == 16000
        assert processor.n_mfcc == 20
        assert processor.max_len == 50

    def test_feature_size_property(self):
        """测试特征维度属性"""
        processor = AudioProcessor(n_mfcc=40, max_len=100)
        assert processor.feature_size == 4000

        processor = AudioProcessor(n_mfcc=20, max_len=50)
        assert processor.feature_size == 1000

    def test_extract_label_from_filename(self):
        """测试从文件名提取标签"""
        processor = AudioProcessor()

        assert processor.extract_label_from_filename("B_ANI01_MC_FN.wav") == "B"
        assert processor.extract_label_from_filename("A_test_file.wav") == "A"
        assert processor.extract_label_from_filename("C_") == "C"

    def test_extract_label_with_custom_delimiter(self):
        """测试自定义分隔符"""
        processor = AudioProcessor()

        assert processor.extract_label_from_filename("label-value-extra", delimiter='-') == "label"

    def test_pad_or_truncate_padding(self):
        """测试特征补零"""
        processor = AudioProcessor(max_len=100)

        features = np.random.randn(40, 50)
        result = processor.pad_or_truncate(features)

        assert result.shape == (40, 100)

    def test_pad_or_truncate_truncation(self):
        """测试特征截断"""
        processor = AudioProcessor(max_len=50)

        features = np.random.randn(40, 100)
        result = processor.pad_or_truncate(features)

        assert result.shape == (40, 50)

    def test_pad_or_truncate_no_change(self):
        """测试无需调整"""
        processor = AudioProcessor(max_len=100)

        features = np.random.randn(40, 100)
        result = processor.pad_or_truncate(features)

        assert result.shape == (40, 100)
        np.testing.assert_array_equal(result, features)

    def test_pad_or_truncate_custom_max_len(self):
        """测试自定义目标长度"""
        processor = AudioProcessor(max_len=100)

        features = np.random.randn(40, 30)
        result = processor.pad_or_truncate(features, max_len=50)

        assert result.shape == (40, 50)

    def test_load_audio_file_not_found(self):
        """测试加载不存在的文件"""
        processor = AudioProcessor()

        with pytest.raises(AudioProcessingError, match="音频文件不存在"):
            processor.load_audio("nonexistent_file.wav")

    @patch('audio_processor.librosa.load')
    def test_load_audio_success(self, mock_load):
        """测试成功加载音频"""
        mock_audio = np.random.randn(22050)
        mock_sr = 22050
        mock_load.return_value = (mock_audio, mock_sr)

        processor = AudioProcessor()
        audio, sr = processor.load_audio("test.wav")

        assert sr == mock_sr
        assert len(audio) == len(mock_audio)
        mock_load.assert_called_once()

    @patch('audio_processor.librosa.load')
    def test_load_audio_empty_file(self, mock_load):
        """测试加载空音频文件"""
        mock_load.return_value = (np.array([]), 22050)

        processor = AudioProcessor()

        with pytest.raises(AudioProcessingError, match="音频文件为空"):
            processor.load_audio("empty.wav")

    @patch('audio_processor.librosa.load')
    def test_extract_mfcc(self, mock_load):
        """测试MFCC特征提取"""
        mock_audio = np.random.randn(22050)
        mock_load.return_value = (mock_audio, 22050)

        processor = AudioProcessor(n_mfcc=40)
        audio, _ = processor.load_audio("test.wav")
        mfccs = processor.extract_mfcc(audio)

        assert mfccs.shape[0] == 40
        mock_load.assert_called_once()

    def test_fit_scaler(self, sample_audio_data):
        """测试scaler拟合"""
        processor = AudioProcessor()

        batch = np.random.randn(10, 4000)
        processor.fit_scaler(batch)

        assert processor.is_fitted
        assert processor._scaler is not None

    def test_transform_features_without_fit(self):
        """测试未拟合时的特征转换"""
        processor = AudioProcessor()

        with pytest.raises(AudioProcessingError, match="Scaler未拟合"):
            processor.transform_features(np.random.randn(10, 4000))

    def test_transform_features_with_fit(self, sample_audio_data):
        """测试拟合后的特征转换"""
        processor = AudioProcessor()

        train_data = np.random.randn(100, 4000)
        processor.fit_scaler(train_data)

        test_data = np.random.randn(10, 4000)
        result = processor.transform_features(test_data)

        assert result.shape == test_data.shape
        assert not np.array_equal(result, test_data)

    def test_save_scaler_without_fit(self, temp_dir):
        """测试未拟合时保存scaler"""
        processor = AudioProcessor()
        save_path = temp_dir / "scaler.pkl"

        with pytest.raises(AudioProcessingError, match="Scaler未拟合"):
            processor.save_scaler(save_path)

    def test_load_scaler_file_not_found(self):
        """测试加载不存在的scaler"""
        processor = AudioProcessor()

        with pytest.raises(AudioProcessingError, match="Scaler文件不存在"):
            processor.load_scaler("nonexistent_scaler.pkl")

    def test_load_scaler_success(self, temp_dir):
        """测试成功加载scaler"""
        import joblib
        from sklearn.preprocessing import StandardScaler

        processor1 = AudioProcessor()
        train_data = np.random.randn(100, 4000)
        processor1.fit_scaler(train_data)

        save_path = temp_dir / "scaler.pkl"
        joblib.dump(processor1._scaler, save_path)

        processor2 = AudioProcessor()
        processor2.load_scaler(save_path)

        assert processor2.is_fitted

        test_data = np.random.randn(5, 4000)
        result1 = processor1.transform_features(test_data)
        result2 = processor2.transform_features(test_data)
        np.testing.assert_array_almost_equal(result1, result2)

    def test_process_audio_workflow(self):
        """测试完整音频处理流程"""
        processor = AudioProcessor(n_mfcc=40, max_len=100)

        with patch('audio_processor.librosa.load') as mock_load:
            mock_audio = np.random.randn(22050)
            mock_load.return_value = (mock_audio, 22050)

            with patch('audio_processor.librosa.feature.mfcc') as mock_mfcc:
                mock_mfcc.return_value = np.random.randn(40, 100)

                features = processor.process_audio("test.wav")

                assert features.shape == (4000,)
                assert isinstance(features, np.ndarray)

    def test_process_batch_workflow(self):
        """测试批量处理流程"""
        processor = AudioProcessor(n_mfcc=40, max_len=100)

        with patch('audio_processor.librosa.load') as mock_load:
            mock_load.return_value = (np.random.randn(22050), 22050)

            with patch('audio_processor.librosa.feature.mfcc') as mock_mfcc:
                mock_mfcc.return_value = np.random.randn(40, 100)

                audio_paths = ["file1.wav", "file2.wav", "file3.wav"]
                result = processor.process_batch(audio_paths)

                assert result.shape == (3, 4000)


class TestLoadAudioProcessor:
    """工厂函数测试"""

    def test_create_without_scaler(self):
        """测试创建无scaler的处理器"""
        processor = load_audio_processor()

        assert processor is not None
        assert not processor.is_fitted

    def test_create_with_scaler_path_not_exist(self):
        """测试加载不存在的scaler"""
        with pytest.raises(AudioProcessingError):
            load_audio_processor(scaler_path="nonexistent.pkl")

    def test_create_with_custom_params(self):
        """测试自定义参数"""
        processor = load_audio_processor(
            sample_rate=16000,
            n_mfcc=20,
            max_len=50
        )

        assert processor.sample_rate == 16000
        assert processor.n_mfcc == 20
        assert processor.max_len == 50


class TestAudioProcessingError:
    """异常类测试"""

    def test_exception_message(self):
        """测试异常消息"""
        error = AudioProcessingError("测试错误")
        assert str(error) == "测试错误"

    def test_exception_inheritance(self):
        """测试异常继承"""
        error = AudioProcessingError("测试")
        assert isinstance(error, Exception)
