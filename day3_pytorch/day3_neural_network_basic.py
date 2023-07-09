#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

# # neuron
# weights = torch.tensor([w1, w2, ..., wn])
# inputs = torch.tensor([x1, x2, ..., xn])
# bias = torch.tensor(b)
# 
# z = torch.dot(weights, inputs) + bias
# 
# # sigmoid
# z = torch.tensor(z)
# output = 1 / (1 + torch.exp(-z))

"""
実装上の注意としては、シグモイド関数は入力値が大きくなると1に近づき、
入力値が小さくなると0に近づく為、数値の安定性にかかわる問題が発生する事が
あります。そのため、特に入力値が大きな正または負の範囲にある場合、数値の
オーバーフローやアンダーフローに注意する必要があります。
PyTorchでは、数値の安定性のためにtorch.sigmoid()関数が提供されており、
この関数を使用することで、数値の範囲を制限しながらシグモイド関数を計算する事が
出来ます。
"""
def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

# テンソルzの作成
z = torch.tensor([0.5, -1.0, 2.0])

# シグモイド関数の適用
output = sigmoid(z)

print(output)

# テンソルzの作成
z = torch.tensor([100.0, -100.0])

# torch.sigmoid()関数の適用

output1 = sigmoid(z)
output2 = torch.sigmoid(z)

print(output1)
print(output2)


"""
2. 損失関数
深層学習では、モデルのパフォーマンスを評価するために損失関数が使用されます。
損失関数は、モデルの予測と真の値の差を表す指標です。目的は損失関数を最小化
するようにモデルのパラメータを調整することです。
代表的な損失関数の1つは平均二乗誤差(Mean Squared Error, MSE)です。
平均二乗誤差は、予測値と真の値の差を二乗し、その平均を計算します。
回帰問題で一般的に使用されます
"""
print("MSELoss")
# 予測値
predictions = torch.tensor([0.8, 0.5, 0.2])

# 真の値
targets = torch.tensor([1.0, 0.7, 0.3])

# 平均二乗誤差の計算
loss_function = nn.MSELoss()
loss = loss_function(predictions, targets)

print(loss)


"""
クロスエントロピー誤差 (Cross Entropy Error)
クロスエントロピー誤差は、主に分類タスクにおいて使用される損失関数です。
予測と真の値の間の交差エントロピーを計算します。
"""
print("CrossEntropyErrorLoss")
# 予測値
predictions = torch.tensor([0.8, 0.5, 0.2])
targets = torch.tensor([1.0, 0.0, 1.0])

loss_function = nn.CrossEntropyLoss()
loss = loss_function(predictions, targets)

print(loss)

"""
バイナリークロスエントロピー誤差 (Binary Cross Entropy Error)
バイナリクロスエントロピー誤差は、二値分類もんだいで用いられる損失関数です。
予測値と真の値の間の交差エントロピーを計算します。
"""
print("BinaryCrossEntropyErrorLoss")
predictions = torch.tensor([0.8, 0.5, 0.2])
targets = torch.tensor([1.0, 0.0, 1.0])

loss_function = nn.BCELoss()
loss = loss_function(predictions, targets)

print(loss)
