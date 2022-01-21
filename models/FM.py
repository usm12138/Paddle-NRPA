#-*-coding:utf-8 -*-
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
default_type = paddle.get_default_dtype()

class PaddleFM(nn.Layer):
    def __init__(self, factor_size: int, fm_k: int):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = paddle.create_parameter([factor_size, fm_k], default_type)
        self.lin = nn.Linear(factor_size, 1)

        
    def forward(self, x):
        out_1 = paddle.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = paddle.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        
        return out
class FactorizationMachine(nn.Layer):

    def __init__(self, factor_size: int, fm_k: int):
        super().__init__()
        self.linear = nn.Linear(factor_size, 1)
        self.v = paddle.create_parameter([factor_size, fm_k], default_type)
        self.drop = nn.Dropout(0.)

    def forward(self, x):
        # linear regression
        # print(1, x.shape)
        w = self.linear(x)
        # print(2, w.shape)
        w = w.squeeze(1)
        # print(3, w.shape)

        # cross feature
        inter1 = paddle.matmul(x, self.v)
        inter2 = paddle.matmul(x**2, self.v**2)
        inter = (inter1**2 - inter2) * 0.5
        inter = self.drop(inter)
        # print(inter.shape, w)
        inter = paddle.sum(inter, axis=1)

        return w + inter
