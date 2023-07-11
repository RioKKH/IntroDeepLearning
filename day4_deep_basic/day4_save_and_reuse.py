#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim


"""
深層学習モデルの学習後には、そのモデルを保存してあとで再利用したり、他の環境で
読み込んで使用したりすることが良くあります。PyTorchでは、学習済みモデルの保存と
読み込みが簡単に行えるようになっています。 
"""
# モデルの定義
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


model = MyModel()

# 学習済みモデルの保存
torch.save(model.state_dict(), 'model.pth')

# 学習済みモデルの読み込み
model = MyModel()
model.load_state_dict(torch.load('model.pth'))

# 新なデータの予測
input_data = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
predictions = model(input_data)
print(predictions)
