#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4. モデルの定義と学習
深層学習では、モデルの定義と学習を行います。モデルの定義では、層を積み重ねて
ニューラルネットワークのアーキテクチャを定義します。学習では、データをモデルに
入力し、パラメータを調整して予測を改善します。
"""

import torch
import torch.nn as nn
import torch.optim as optim

# データの準備
inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
targets = torch.tensor([3.0, 4.0, 5.0])

# モデルの定義
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(2, 5) # 入力次元: 2, 出力次元: 5
        self.fc2 = nn.Linear(5, 1) # 入力次元: 5, 出力次元: 1

    def forward(self, x):
        """
        nn.Moduleクラスを継承したモデルクラスでforward()メソッドを定義する事で、
        モデルが入力データを受け取り、予測値を出力する処理を記述します。
        モデルのインスタンスを作成し、そのインスタンスを関数のように呼び出すと、
        内部的にfoward()メソッドが呼び出されて、順伝搬処理が実行されます。
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルのインスタンス化
model = MyModel()

# 損失関数と最適化アルゴリズムの定義
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 学習のループ
for epoch in range(100):
    # 勾配を初期化する
    optimizer.zero_grad()

    # モデルの予測値を計算
    predictions = model(inputs)

    # 損失を計算する
    loss = loss_function(predictions.squeeze(), targets)

    # バックプロパゲーションとパラメータの更新
    loss.backward()
    optimizer.step()

# 学習後のパラメータを表示する
print(model.state_dict())
# fc1.weight: 入力層から隠れ層1への重み行列 (shape: (5, 2))
# fc1.bias: 隠れ層1のバイアスベクトル (shape: (5, ))
# fc2.weight: 隠れ層1から出力層への重み行列 (shape: (1, 5))
# fc2.bias: 出力層のバイアスベクトル (shape: (1,))
# OrderedDict([
#     ('fc1.weight', tensor([[ 0.1195, -0.2768],
#                            [-0.6200, -0.2808],
#                            [-0.4699, -0.3335],
#                            [ 0.0966, -0.3447],
#                            [ 0.3400,  0.7896]])),
#     ('fc1.bias', tensor([ 0.0793, -0.5589,  0.3335, -0.2467,  0.8648])),
#     ('fc2.weight', tensor([[-0.2480, -0.4228,  0.2780,  0.1000,  0.9961]])),
#     ('fc2.bias', tensor([0.0756]))])            
