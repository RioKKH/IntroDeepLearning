#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

# モデルの定義
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel(), self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

model = MyModel()
