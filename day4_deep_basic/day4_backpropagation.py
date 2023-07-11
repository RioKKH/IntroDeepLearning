#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深層学習では、モデルの学習を行うためにバックプロパゲーションと勾配降下法という
手法が使われる。バックプロパゲーション(Backpropagation)は、誤差を出力層から
逆方向に伝播させながら、各層のパラメータに対する勾配を計算する手法です。
これにより、各パラメータが損失関数の勾配に従って更新されます。
勾配降下法 (Gradient Descent)は、バックプロパゲーションで計算された勾配を使って
モデルのパラメータを更新する手法です。パラメータの更新は、現在のパラメータから
学習率を掛けた勾配の逆方向に一定のステップサイズで行います。
これにより、損失関数の値を最小化する方向にパラメータを調整します。
"""

import torch
import torch.nn as nn
import torch.optim as optim

# 入力データと真の値
inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
targets = torch.tensor([3.0, 4.0, 5.0])

# モデルの定義
# 線形モデルをあらわすnn.Linearを利用してモデルを定義している
# Linear(入力の次元数, 出力の次元数)の線形変換を行う事を意味する。
model = nn.Linear(2, 1)

# 損失関数と最適化アルゴリズムの定義
# 平均二乗誤差を計算する
loss_function = nn.MSELoss()
# 最適化アルゴリズムとして確率的勾配降下法を使用。学習率を0.01としている
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 学習のループ
for epoch in range(100):
    # 勾配を初期化
    optimizer.zero_grad()

    # モデルの予測値を計算
    predictions = model(inputs).squeeze()

    # 損失を計算
    loss = loss_function(predictions, targets)

    # バックプロパゲーションとパラメータの更新
    loss.backward()
    optimizer.step()

# 学習後のパラメータを表示
print(model.weight)
print(model.bias)
"""
結果は以下のようになります。

Parameter containing:
tensor([[0.0309, 1.3043]], requires_grad=True)
Parameter containing:
tensor([-0.0553], requires_grad=True)               
"""
