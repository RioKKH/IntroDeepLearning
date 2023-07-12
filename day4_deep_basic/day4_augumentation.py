#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
データの前処理と拡張：
データの前処理や拡張を行って、モデルｎ学習に有益な情報を追加します。
以下はデータの正規化とデータの水増し（ランダムな回転と反転）を行う例です。
"""
import torch
import torchvision.transforms as transforms

# データの正規化と水増しの定義
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip()
])

# データセットの作成
dataset = YourDataset(transform=data_transforms)
