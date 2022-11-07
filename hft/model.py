import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, batch_first, fc_layer_num, *fc_layer_param) -> None:
        super().__init__()
        """
        input_size: 输入数据的大小, 也就是前面例子中每个单词向量的长度
        hidden_size: 隐藏层的大小(即隐藏层节点数量), 输出向量的维度等于隐藏节点数
        num_layers: recurrent layer的数量,默认等于1.
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        # self.num_output = num_output
        self.batch_first = batch_first
        self.batch_size = batch_size
        # self.dropout = dropout
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc_layer_list = []
        self.fc1 = nn.Linear(hidden_size, fc_layer_param[0])
        self.fc_layer_list.append(self.fc1)
        for i in range(1, fc_layer_num):
            current_fc = nn.Linear(fc_layer_param[i-1], fc_layer_param[i])
            self.fc_layer_list.append(current_fc)
        

    def forward(self, x):
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # print(f"out0 size {out.size()}")
        # print(f"hn size {hn.size()}")
        # print(f"cn size {cn.size()}")

        # LSTM + Attention (Weighted average of the output)
        w_omiga = torch.randn(out.size(0), self.hidden_size, 1)
        H = torch.nn.Tanh()(out)
        weights = torch.nn.Softmax(dim=-1)(torch.bmm(H, w_omiga).squeeze()).unsqueeze(dim=-1).repeat(1,1,self.hidden_size)
        atten_output = torch.mul(out, weights)
        atten_output = atten_output.sum(dim=-2)

        out = self.fc_layer_list[0](atten_output)
        for fc_layer in self.fc_layer_list[1:]:
            out = fc_layer(out)
        
        return out

    

