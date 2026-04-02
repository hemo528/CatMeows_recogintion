import os
import librosa
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集目录
dataset_dir = 'dataset/dataset'

# 创建结果目录
if not os.path.exists('result/best'):
    os.makedirs('result/best')
if not os.path.exists('result/latest'):
    os.makedirs('result/latest')

# 读取数据集文件名
files = os.listdir(dataset_dir)

# 初始化数据框
data = {'filename': [], 'background': [], 'features': []}

# 解析文件名并提取特征
max_len = 100  # 设置最大长度，这里假设为100
for filename in files:
    # 解析文件名以获取背景标签
    background = filename.split('_')[0]

    # 加载音频文件并提取特征
    filepath = os.path.join(dataset_dir, filename)
    audio, sr = librosa.load(filepath, sr=None)  # 加载音频数据
    # 提取MFCC特征
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # 调整特征的长度
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]

    # 将数据添加到数据框
    data['filename'].append(filename)
    data['background'].append(background)
    data['features'].append(mfccs.flatten())

# 转换成DataFrame
df = pd.DataFrame(data)

# 对背景标签进行编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['background'].values)

# 划分训练集和测试集
X = np.array(df['features'].tolist())
y = y_encoded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建PyTorch数据集
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = AudioDataset(X_train_scaled, y_train)
test_dataset = AudioDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义神经网络模型
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(max_len*40, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, len(np.unique(y)))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = AudioClassifier().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化最佳准确率
best_accuracy = 0.0

# 训练模型
epochs = 5000
for epoch in range(epochs):
    model.train()
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(features.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device).long()
            outputs = model(features.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy}%')

    # 保存最新模型
    torch.save(model.state_dict(), 'result/latest/model_latest.pth')#torch.save(model, 'result/latest/model_latest.pth')

    # 如果当前模型是最佳模型，则保存
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'result/best/model_best.pth')#torch.save(model, 'result/best/model_best.pth')
        print(f'最新的模型: {best_accuracy}%')

print(f'最高的准确率: {best_accuracy}%')
