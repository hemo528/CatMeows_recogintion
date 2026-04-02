"""
神经网络模型模块测试
测试模型架构和工厂函数
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile

from model import (
    AudioClassifier,
    CNN1DClassifier,
    ModelFactory,
    save_model,
    load_model
)


class TestAudioClassifier:
    """AudioClassifier测试"""

    def test_init_default_params(self):
        """测试默认参数初始化"""
        model = AudioClassifier(input_size=4000, num_classes=3)

        assert model.input_size == 4000
        assert model.num_classes == 3
        assert model.hidden_sizes == [512, 256, 64]
        assert model.dropout_rate == 0.0

    def test_init_custom_params(self):
        """测试自定义参数"""
        model = AudioClassifier(
            input_size=1000,
            num_classes=5,
            hidden_sizes=[256, 128],
            dropout_rate=0.3
        )

        assert model.input_size == 1000
        assert model.num_classes == 5
        assert model.hidden_sizes == [256, 128]
        assert model.dropout_rate == 0.3

    def test_forward_pass(self):
        """测试前向传播"""
        model = AudioClassifier(input_size=4000, num_classes=3)
        batch_size = 16
        x = torch.randn(batch_size, 4000)

        output = model(x)

        assert output.shape == (batch_size, 3)

    def test_forward_single_sample(self):
        """测试单个样本前向传播"""
        model = AudioClassifier(input_size=4000, num_classes=3)
        x = torch.randn(1, 4000)

        output = model(x)

        assert output.shape == (1, 3)

    def test_predict_proba(self):
        """测试概率预测"""
        model = AudioClassifier(input_size=4000, num_classes=3)
        x = torch.randn(16, 4000)

        probs = model.predict_proba(x)

        assert probs.shape == (16, 3)
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-5)

    def test_predict(self):
        """测试类别预测"""
        model = AudioClassifier(input_size=4000, num_classes=3)
        x = torch.randn(16, 4000)

        predictions = model.predict(x)

        assert predictions.shape == (16,)
        assert torch.all((predictions >= 0) & (predictions < 3))

    def test_predict_proba_mode(self):
        """测试预测时模型模式"""
        model = AudioClassifier(input_size=4000, num_classes=3, dropout_rate=0.5)
        model.train()

        x = torch.randn(16, 4000)

        probs_train = model.predict_proba(x)

        model.eval()
        probs_eval = model.predict_proba(x)

        assert not torch.allclose(probs_train, probs_eval, atol=1e-1)

    def test_get_num_parameters(self):
        """测试参数数量计算"""
        model = AudioClassifier(input_size=4000, num_classes=3)
        num_params = model.get_num_parameters()

        assert num_params > 0
        assert isinstance(num_params, int)

    def test_get_num_trainable_parameters(self):
        """测试可训练参数数量"""
        model = AudioClassifier(input_size=4000, num_classes=3)
        num_trainable = model.get_num_trainable_parameters()

        assert num_trainable > 0
        assert num_trainable <= model.get_num_parameters()

    def test_parameters_frozen(self):
        """测试冻结参数"""
        model = AudioClassifier(input_size=4000, num_classes=3)

        for param in model.parameters():
            param.requires_grad = False

        num_trainable = model.get_num_trainable_parameters()
        assert num_trainable == 0

    def test_weight_initialization(self):
        """测试权重初始化"""
        model = AudioClassifier(input_size=100, num_classes=2)

        for name, param in model.named_parameters():
            if 'weight' in name:
                assert not torch.all(param == 0)
            elif 'bias' in name:
                assert torch.all(param == 0)


class TestCNN1DClassifier:
    """CNN1DClassifier测试"""

    def test_init(self):
        """测试初始化"""
        model = CNN1DClassifier(
            input_size=4000,
            num_classes=3,
            n_mfcc=40
        )

        assert model.input_size == 4000
        assert model.num_classes == 3
        assert model.n_mfcc == 40

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        model = CNN1DClassifier(input_size=4000, num_classes=3, n_mfcc=40)
        batch_size = 8
        x = torch.randn(batch_size, 4000)

        output = model(x)

        assert output.shape == (batch_size, 3)

    def test_predict_proba(self):
        """测试概率预测"""
        model = CNN1DClassifier(input_size=4000, num_classes=3, n_mfcc=40)
        x = torch.randn(8, 4000)

        probs = model.predict_proba(x)

        assert probs.shape == (8, 3)
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)


class TestModelFactory:
    """模型工厂测试"""

    def test_create_fc_model(self):
        """测试创建全连接模型"""
        model = ModelFactory.create(
            model_type='fc',
            input_size=4000,
            num_classes=3
        )

        assert isinstance(model, AudioClassifier)
        assert model.input_size == 4000

    def test_create_cnn1d_model(self):
        """测试创建CNN模型"""
        model = ModelFactory.create(
            model_type='cnn1d',
            input_size=4000,
            num_classes=3
        )

        assert isinstance(model, CNN1DClassifier)

    def test_create_unknown_type(self):
        """测试创建未知类型模型"""
        with pytest.raises(ValueError, match="未知的模型类型"):
            ModelFactory.create(
                model_type='unknown',
                input_size=4000,
                num_classes=3
            )

    def test_create_with_custom_params(self):
        """测试创建带自定义参数的模型"""
        model = ModelFactory.create(
            model_type='fc',
            input_size=1000,
            num_classes=5,
            hidden_sizes=[256, 128, 64],
            dropout_rate=0.3
        )

        assert model.hidden_sizes == [256, 128, 64]
        assert model.dropout_rate == 0.3

    def test_register_new_model(self):
        """测试注册新模型类型"""

        class CustomModel(nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc = nn.Linear(input_size, num_classes)

            def forward(self, x):
                return self.fc(x)

        ModelFactory.register('custom', CustomModel)

        model = ModelFactory.create(
            model_type='custom',
            input_size=100,
            num_classes=5
        )

        assert isinstance(model, CustomModel)

    def test_list_models(self):
        """测试列出所有模型类型"""
        models = ModelFactory.list_models()

        assert 'fc' in models
        assert 'cnn1d' in models
        assert isinstance(models, list)


class TestSaveLoadModel:
    """模型保存加载测试"""

    def test_save_model_basic(self, temp_dir):
        """测试基本模型保存"""
        model = AudioClassifier(input_size=4000, num_classes=3)
        save_path = temp_dir / "model.pth"

        save_model(model, str(save_path))

        assert save_path.exists()

    def test_save_model_with_optimizer(self, temp_dir):
        """测试带优化器的模型保存"""
        model = AudioClassifier(input_size=4000, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters())
        save_path = temp_dir / "model_with_opt.pth"

        save_model(
            model,
            str(save_path),
            optimizer=optimizer,
            epoch=10,
            metrics={'accuracy': 0.95}
        )

        assert save_path.exists()

    def test_load_model_basic(self, temp_dir):
        """测试基本模型加载"""
        model1 = AudioClassifier(input_size=4000, num_classes=3)

        x = torch.randn(4, 4000)
        output1 = model1(x)

        save_path = temp_dir / "model.pth"
        save_model(model1, str(save_path))

        model2 = load_model(str(save_path), model_type='fc')

        model1.eval()
        model2.eval()

        output2 = model2(x)

        assert model2.input_size == 4000
        assert model2.num_classes == 3
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_load_model_custom_config(self, temp_dir):
        """测试加载自定义配置模型"""
        model = AudioClassifier(
            input_size=1000,
            num_classes=5,
            hidden_sizes=[256, 128]
        )

        save_path = temp_dir / "model_custom.pth"
        save_model(model, str(save_path))

        loaded_model = load_model(
            str(save_path),
            model_type='fc'
        )

        assert loaded_model.input_size == 1000
        assert loaded_model.num_classes == 5
        assert loaded_model.hidden_sizes == [256, 128]

    def test_load_nonexistent_model(self):
        """测试加载不存在的模型"""
        with pytest.raises(EOFError):
            load_model("nonexistent_model.pth")


class TestModelIntegration:
    """模型集成测试"""

    def test_training_loop(self, sample_audio_batch):
        """测试完整训练循环"""
        features, labels = sample_audio_batch

        model = AudioClassifier(input_size=4000, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        features_tensor = torch.FloatTensor(features)
        labels_tensor = torch.LongTensor(labels)

        initial_loss = None
        for _ in range(10):
            optimizer.zero_grad()
            outputs = model(features_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert final_loss < initial_loss

    def test_model_save_load_inference(self, temp_dir):
        """测试模型保存加载推理流程"""
        model = AudioClassifier(input_size=4000, num_classes=3)
        model.eval()

        test_input = torch.randn(4, 4000)
        original_output = model(test_input)

        save_path = temp_dir / "inference_model.pth"
        save_model(model, str(save_path))

        loaded_model = load_model(str(save_path), model_type='fc')
        loaded_model.eval()

        loaded_output = loaded_model(test_input)

        assert torch.allclose(original_output, loaded_output, atol=1e-6)

    def test_multi_model_comparison(self):
        """测试多个模型比较"""
        fc_model = AudioClassifier(input_size=4000, num_classes=3)
        cnn_model = ModelFactory.create(
            model_type='cnn1d',
            input_size=4000,
            num_classes=3
        )

        test_input = torch.randn(4, 4000)

        fc_output = fc_model(test_input)
        cnn_output = cnn_model(test_input)

        assert fc_output.shape == cnn_output.shape == (4, 3)

        fc_probs = fc_model.predict_proba(test_input)
        cnn_probs = cnn_model.predict_proba(test_input)

        assert torch.allclose(fc_probs.sum(dim=1), torch.ones(4), atol=1e-5)
        assert torch.allclose(cnn_probs.sum(dim=1), torch.ones(4), atol=1e-5)
