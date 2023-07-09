#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd


# データフレームの生成
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

print(df)
print(type(df))

# 列Ａを選択する
print(df['A'])

# 最初の行を選択
print(df.loc[0])
# 最初の行を選択する
# ilocは整数の位置に基づいて行を選択する。
    # locがラベルに基づいて行を選択するのに対して、ilocは位置に基づいて行を選択する
print("iloc")
print(df.iloc[0])

# 最初の行、列'A'の要素を変更
df.loc[0, 'A'] = 10
print(df)
# ilocを使って特定の要素を選択し、新しい値を代入する方法を見る
# 最初の行、最初の列の要素を変更する
df.iloc[0, 0] = 20
print(df)

# Pandasを使ってデータフレームの基本的な統計量を計算する方法を見る
# 列Ａの平均値
mean = df['A'].mean()
print(mean)

# 列Aの最大値
max_value = df['A'].max()
print(max_value)

# 列Aの最小値
min_value = df['A'].min()
print(min_value)
