#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# Numpy配列の作成
a = np.array([1, 2, 3, 4, 5])
print(a)
# aの型を確認する
print(type(a))

# 配列の形状
print(a.shape)

# 配列の次元数
print(a.ndim)

# 配列の最初の要素
print(a[0])

# 配列の最後の要素
print(a[-1])

# 配列の最初の要素を変更する
a[0] = 10
print(a)

# 配列aの要素のうち、3より大きい要素を選択する
b = a[a > 3]
print(b)

# 配列aのすべての要素を2倍する。これはベクトル化と呼ばれる
c = a * 2
print(c)

# 配列aの平均値
mean = a.mean()
print(mean)

# 配列aの最大値
max_value = a.max()
print(max_value)

# 配列aの最小値
min_value = a.min()
print(min_value)

