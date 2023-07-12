#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        # fcはFully Connected Layerで全結合層を意味する
        # 下のコードは  fc1は入力層の次元数(input_size)から中間層の次元数
        # (hidden_size)への全結合層を表し、fc2は中間層の次元数から出力の
        # 次元数(output_size)への全結合層を表します。
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hiden_size, output_size)

    def forward(self, x):
        """
        ReLUの__call__を使ったか書き方について
        オブジェクトを関数のように呼び出すことは、以下のような利点があります。

        1. **簡潔な記述**: オブジェクトを関数のように呼び出すことで、
        より短く簡潔なコードを書くことができます。例えば、`nn.ReLU()(x)`のように
        書くことで、ReLU関数を作成して適用する一連の手順を1行で表現できます。

        2. **柔軟な操作**: オブジェクトを関数のように呼び出すことで、より柔軟な
        操作が可能になります。オブジェクトが関数として呼び出される際に、
        内部の`__call__()`メソッドが実行されるため、より高度な処理やパラメータの
        調整が可能です。

        3. **再利用性**: オブジェクトを関数のように呼び出すことで、再利用性が
        向上します。関数として呼び出せるオブジェクトは、他の関数やモジュールと
        同じように扱えるため、再利用や組み合わせが容易になります。

        4. **パイプライン処理**: オブジェクトを関数のように呼び出すことで、
        パイプライン処理を簡潔に表現できます。複数のオブジェクトを順番に
        呼び出して処理を行う場合、関数のような記述方法を使用することで、
        処理の流れを直感的に表現することができます。

        以上のような理由から、オブジェクトを関数のように呼び出す表現方法は、
        コードの可読性や柔軟性を向上させるのに役立ちます。また、PyTorchでは
        モジュールや関数のインスタンスを関数のように呼び出すことが多く、
        この表現方法が広く利用されています。
        """
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


model = Mymodel(input_size=10, hidden_size=20, output_size=5)

