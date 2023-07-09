#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

# テンソルの作成
x = torch.tensor([1, 2, 3, 4, 5])
print(x)
print(type(x))

# テンソルの形状を表示
print(x.shape) # --> tensor([1, 2, 3, 4, 5])

# テンソルの次元数を表示
print(x.ndim) # --> 1

# テンソルの要素数を表示
print(x.numel()) # --> 5

# テンソルのインデックスを使った要素へのアクセス
print(x[0]) # --> tensor(1)

# テンソルの演算
y = x + 2
print(y) # --> tensor([3, 4, 5, 6, 7])

z = torch.sin(x) # --> tensor([ 0.8415,  0.9093,  0.1411, -0.7568, -0.9589])
print(z)

# テンソルの形状変更
x = torch.tensor([1, 2, 3, 4, 5])
y = x.view(5, 1)
print(y, y.shape) # torch.Size([5, 1])

# テンソルに次元追加 z = x.unsqueeze(0)
print(x, x.shape) # torch.Size([5])
print(z, z.shape) # torch.Size([1, 5])
