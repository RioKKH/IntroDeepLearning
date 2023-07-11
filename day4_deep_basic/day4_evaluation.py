#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深層学習モデルを構築し、学習を行った後は、モデルの性能を評価する必要があります。
モデルの評価は、テストデータセットや検証データセットを利用して行われます。
"""

import torch
import torch.nn as nn

# モデルの定義
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


# 学習済みモデルの読み込み
model = MyModel()
model.load_state_dict(torch.load('model.pth'))

# テストデータの準備
test_inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
test_targets = torch.tensor([3.0, 4.0])

# モデルの評価
model.eval() 
# モデルを評価モードに設定。
# 勾配計算を無効化。評価時には勾配の計算が不要なため
with torch.no_grad():
    predictions = model(test_inputs)
    loss_function = nn.MSELoss()
    # 得られた予測結果と正解データを用いて、適切な評価しよう（この場合は
    # 平均二乗誤差）を計算します。
    loss = loss_function(predictions.squeeze(), test_targets)

print('Test Loss', loss.item())
