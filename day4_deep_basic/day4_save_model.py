#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
9. モデルの保存と読み込み

1. モデルの保存：torch.save()関数を利用して学習済みモデルをほぞなします。

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

# モデルの保存
torch.save(model.state_dict(), 'model9.pth')
print("モデルの保存完了")

# モデルの読み込み
loaded_model = MyModel()
loaded_model.load_state_dict(torch.load('model9.pth'))
print("モデルの読み込み完了")

# モデルの再評価
loaded_model.eval()


