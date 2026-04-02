"""
检测模块测试
测试Detector类和预测功能
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import pickle

from detect_refactored import (
    Detector,
    DetectorFactory,
    PredictionResult,
    predict_audio,
    predict_directory
)


class TestPredictionResult:
    """预测结果数据类测试"""

    def test_init(self):
        """测试初始化"""
        result = PredictionResult(
            audio_path="test.wav",
            predicted_class=0,
            predicted_label="烦躁",
            confidence=0.85,
            probabilities={"烦躁": 0.85, "饥饿": 0.10, "不安": 0.05}
        )

        assert result.audio_path == "test.wav"
        assert result.predicted_class == 0
        assert result.predicted_label == "烦躁"
        assert result.confidence == 0.85

    def test_str_representation(self):
        """测试字符串表示"""
        result = PredictionResult(
            audio_path="test.wav",
            predicted_class=1,
            predicted_label="饥饿",
            confidence=0.92,
            probabilities={"烦躁": 0.05, "饥饿": 0.92, "不安": 0.03}
        )

        str_result = str(result)

        assert "test.wav" in str_result
        assert "饥饿" in str_result
        assert "1" in str_result
        assert "92.00%" in str_result


class TestDetector:
    """Detector类测试"""

    @pytest.fixture
    def mock_detector(self, mock_model, mock_audio_processor):
        """创建模拟检测器"""
        label_encoder = Mock()
        label_encoder.inverse_transform.return_value = ["烦躁"]

        detector = Detector(
            model=mock_model,
            audio_processor=mock_audio_processor,
            label_encoder=label_encoder,
            emotion_labels={0: '烦躁', 1: '饥饿', 2: '不安'},
            device='cpu'
        )
        return detector

    def test_init(self, mock_model, mock_audio_processor):
        """测试初始化"""
        label_encoder = Mock()

        detector = Detector(
            model=mock_model,
            audio_processor=mock_audio_processor,
            label_encoder=label_encoder,
            device='cpu'
        )

        assert detector.model is not None
        assert detector.audio_processor is not None
        assert detector.label_encoder is not None
        assert detector.device == 'cpu'

    def test_init_default_emotion_labels(self, mock_model, mock_audio_processor):
        """测试默认情绪标签"""
        label_encoder = Mock()

        detector = Detector(
            model=mock_model,
            audio_processor=mock_audio_processor,
            label_encoder=label_encoder
        )

        assert detector.emotion_labels[0] == '烦躁'
        assert detector.emotion_labels[1] == '饥饿'
        assert detector.emotion_labels[2] == '不安'

    def test_init_custom_emotion_labels(self, mock_model, mock_audio_processor):
        """测试自定义情绪标签"""
        label_encoder = Mock()
        custom_labels = {0: 'happy', 1: 'sad', 2: 'neutral'}

        detector = Detector(
            model=mock_model,
            audio_processor=mock_audio_processor,
            label_encoder=label_encoder,
            emotion_labels=custom_labels
        )

        assert detector.emotion_labels == custom_labels

    def test_predict_single_file_not_found(self, mock_detector):
        """测试文件不存在时的预测"""
        with pytest.raises(FileNotFoundError, match="音频文件不存在"):
            mock_detector.predict_single("nonexistent.wav")

    @patch('detect_refactored.Path.exists')
    def test_predict_single_success(self, mock_exists, mock_detector):
        """测试成功预测单个文件"""
        mock_exists.return_value = True

        mock_detector.audio_processor.process_audio = Mock(
            return_value=np.random.randn(4000)
        )
        mock_detector.audio_processor.transform_features = Mock(
            return_value=np.random.randn(1, 4000)
        )

        result = mock_detector.predict_single("test.wav")

        assert isinstance(result, PredictionResult)
        assert result.predicted_class in [0, 1, 2]
        assert result.audio_path == "test.wav"

    def test_predict_batch_empty(self, mock_detector):
        """测试批量预测空列表"""
        results = mock_detector.predict_batch([])
        assert results == []

    @patch('detect_refactored.Path.exists')
    def test_predict_batch_multiple_files(self, mock_exists, mock_detector):
        """测试批量预测多个文件"""
        mock_exists.return_value = True

        mock_detector.audio_processor.process_audio = Mock(
            return_value=np.random.randn(4000)
        )
        mock_detector.audio_processor.transform_features = Mock(
            return_value=np.random.randn(1, 4000)
        )

        audio_paths = ["file1.wav", "file2.wav", "file3.wav"]
        results = mock_detector.predict_batch(audio_paths, show_progress=False)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, PredictionResult)

    @patch('detect_refactored.Path.exists')
    def test_predict_batch_with_failure(self, mock_exists, mock_detector):
        """测试批量预测中的部分失败"""
        mock_exists.return_value = True

        call_count = [0]
        original_process = mock_detector.audio_processor.process_audio

        def process_with_error(path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("模拟错误")
            return np.random.randn(4000)

        mock_detector.audio_processor.process_audio = Mock(
            side_effect=process_with_error
        )
        mock_detector.audio_processor.transform_features = Mock(
            return_value=np.random.randn(1, 4000)
        )

        audio_paths = ["file1.wav", "file2.wav", "file3.wav"]
        results = mock_detector.predict_batch(audio_paths, show_progress=False)

        assert len(results) == 2

    def test_predict_directory_not_found(self, mock_detector):
        """测试目录不存在"""
        with pytest.raises(FileNotFoundError, match="目录不存在"):
            mock_detector.predict_directory("nonexistent_dir")

    @patch('detect_refactored.Path.exists')
    @patch('detect_refactored.Path.is_dir')
    @patch('detect_refactored.Path.glob')
    def test_predict_directory_no_files(
        self,
        mock_glob,
        mock_is_dir,
        mock_exists,
        mock_detector
    ):
        """测试目录下没有匹配文件"""
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_glob.return_value = []

        results = mock_detector.predict_directory("empty_dir")

        assert results == []


class TestDetectorFactory:
    """检测器工厂测试"""

    def test_from_config_missing_files(self):
        """测试缺少文件时的工厂方法"""
        with pytest.raises(Exception):
            DetectorFactory.from_config(
                model_path="nonexistent_model.pth",
                scaler_path="nonexistent_scaler.pkl",
                label_encoder_path="nonexistent_encoder.pkl"
            )


class TestPredictFunctions:
    """预测快捷函数测试"""

    @patch('detect_refactored.DetectorFactory.from_config')
    def test_predict_audio_function(self, mock_factory):
        """测试快捷预测函数"""
        mock_detector = Mock()
        mock_result = PredictionResult(
            audio_path="test.wav",
            predicted_class=0,
            predicted_label="烦躁",
            confidence=0.9,
            probabilities={}
        )
        mock_detector.predict_single.return_value = mock_result
        mock_factory.return_value = mock_detector

        result = predict_audio("test.wav")

        assert isinstance(result, PredictionResult)
        mock_factory.assert_called_once()
        mock_detector.predict_single.assert_called_once_with("test.wav")

    @patch('detect_refactored.DetectorFactory.from_config')
    def test_predict_directory_function(self, mock_factory):
        """测试目录预测函数"""
        mock_detector = Mock()
        mock_results = [
            PredictionResult("file1.wav", 0, "烦躁", 0.9, {}),
            PredictionResult("file2.wav", 1, "饥饿", 0.8, {})
        ]
        mock_detector.predict_directory.return_value = mock_results
        mock_factory.return_value = mock_detector

        results = predict_directory("test_dir")

        assert len(results) == 2
        mock_factory.assert_called_once()
        mock_detector.predict_directory.assert_called_once()


class TestDetectorIntegration:
    """检测器集成测试"""

    def test_full_prediction_pipeline(self, trained_model, mock_audio_processor):
        """测试完整预测流程"""
        label_encoder = Mock()
        label_encoder.inverse_transform.return_value = ["烦躁"]

        mock_audio_processor.is_fitted = True
        mock_audio_processor.fit_scaler(np.random.randn(100, 4000))
        mock_audio_processor.process_audio = Mock(
            return_value=np.random.randn(4000)
        )
        mock_audio_processor.transform_features = Mock(
            return_value=np.random.randn(1, 4000)
        )

        detector = Detector(
            model=trained_model,
            audio_processor=mock_audio_processor,
            label_encoder=label_encoder,
            device='cpu'
        )

        with patch('detect_refactored.Path.exists', return_value=True):
            result = detector.predict_single("test.wav")

        assert isinstance(result, PredictionResult)
        assert result.predicted_class in [0, 1, 2]
        assert 0 <= result.confidence <= 1
        assert len(result.probabilities) == 3

    def test_model_inference_consistency(self, trained_model, mock_audio_processor):
        """测试模型推理一致性"""
        label_encoder = Mock()
        label_encoder.inverse_transform.return_value = ["烦躁"]

        mock_audio_processor.is_fitted = True
        mock_audio_processor.fit_scaler(np.random.randn(100, 4000))
        mock_audio_processor.process_audio = Mock(
            return_value=np.random.randn(4000)
        )
        mock_audio_processor.transform_features = Mock(
            return_value=np.random.randn(1, 4000)
        )

        detector = Detector(
            model=trained_model,
            audio_processor=mock_audio_processor,
            label_encoder=label_encoder,
            device='cpu'
        )

        with patch('detect_refactored.Path.exists', return_value=True):
            result1 = detector.predict_single("test.wav")

        with patch('detect_refactored.Path.exists', return_value=True):
            result2 = detector.predict_single("test.wav")

        assert result1.predicted_class == result2.predicted_class
