#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
8. モデルの評価と推論

深層学習モデルの構築や学習が完了したら、モデルの評価と推論を行うことが重要です。
以下にモデルの評価と推論の手順を示します。
1. モデルの評価：モデルの性能を評価するために、テストデータセットや検証データ
セットを用いてモデルの精度や損失を計算します。一般的な評価指標としては、正解率、
損失関数（例：クロスエントロピー誤差）、混同行列などがあります。
2. 推論の実行：モデルを使用して新しいデータに対して推論を行います。推論では、
未知のデータに対してモデルの予測結果を生成します。通常、テストデータセットや
実際のデータを用いて推論を行います。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# モデルの定義
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


# データの準備
train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset  = MNIST(root='./data', train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader  = DataLoader(test_dataset,  batch_size=64, shuffle=True)

# モデルの初期化
model = MyModel()

# 損失関数とオプティマイザの定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 学習ループ
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# モデルの評価
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


