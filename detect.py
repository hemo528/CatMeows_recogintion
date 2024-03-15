import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 定义音频处理函数
def process_audio(audio_path, max_len=100):
    # 加载音频文件并提取MFCC特征
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # 调整特征的长度
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]

    return mfccs.flatten()

# 加载模型
class AudioClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 加载模型的状态字典
model_path = 'result/best/model_best.pth'
state_dict = torch.load(model_path)

# 初始化模型并加载状态字典
input_size = 40 * 100  # 根据训练时使用的特征长度确定
output_size = 3  # 根据数据集中的类别数量确定，这里是背景标签的类别数量
model = AudioClassifier(input_size, output_size)
model.load_state_dict(state_dict)
model.eval()

# 加载音频文件并进行预
# 测
audio_path = 'wav/2.wav'  # 替换为你自己的音频文件路径
features = process_audio(audio_path)
scaler = StandardScaler()
features_scaled = scaler.fit_transform([features])  # 标准化特征
inputs = torch.tensor(features_scaled, dtype=torch.float32)

# 使用模型进行预测
with torch.no_grad():
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()

# 输出预测结果
if predicted_class == 0:
    print("预测结果：状态为烦躁。")
elif predicted_class == 1:
    print("预测结果：状态为饥饿。")
else:
    print("预测结果：状态为不安。")
