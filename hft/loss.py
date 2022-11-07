import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn



"""
Apart from the existing loss functions: nn.L1Loss, nn.MSELoss, nnHuberLoss,
we have customized loss functions: CosineSimilarityLoss
"""


class PearsonCorrLoss(nn.Module):
    def forward(self, tensor_x, tensor_y):
        x, y, = tensor_x.reshape(-1), tensor_y.reshape(-1)
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        corr = (torch.sum((x - x_mean) * (y - y_mean))) / (torch.sqrt(torch.sum((x - x_mean) ** 2)) * torch.sqrt(torch.sum((y - y_mean) ** 2)))
        return -abs(corr)



class CosineSimilarityLoss(nn.Module):
    def forward(self, tensor_x, tensor_y):
        normalized_x = tensor_x / tensor_x.norm()
        normalized_y = tensor_y / tensor_y.norm()
        cossim = (normalized_x * normalized_y).sum()
        return -abs(cossim)





class ProjectedDotProductSimilarity(nn.Module):
   
    def __init__(self, tensor_1_dim, tensor_2_dim, projected_dim,
                 reuse_weight=False, bias=False, activation=None):
        super(ProjectedDotProductSimilarity, self).__init__()
        self.reuse_weight = reuse_weight
        self.projecting_weight_1 = nn.Parameter(torch.Tensor(tensor_1_dim, projected_dim))
        if self.reuse_weight:
            if tensor_1_dim != tensor_2_dim:
                raise ValueError('if reuse_weight=True, tensor_1_dim must equal tensor_2_dim')
        else:
            self.projecting_weight_2 = nn.Parameter(torch.Tensor(tensor_2_dim, projected_dim))
        self.bias = nn.Parameter(torch.Tensor(1)) if bias else None
        self.activation = activation
 
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.projecting_weight_1)
        if not self.reuse_weight:
            nn.init.xavier_uniform_(self.projecting_weight_2)
        if self.bias is not None:
            self.bias.data.fill_(0)
 
    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = torch.matmul(tensor_1, self.projecting_weight_1)
        if self.reuse_weight:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_1)
        else:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_2)
        result = (projected_tensor_1 * projected_tensor_2).sum(dim=-1)
        if self.bias is not None:
            result = result + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result



class BiLinearSimilarity(nn.Module):
 
    def __init__(self, tensor_1_dim, tensor_2_dim, activation=None):
        super(BiLinearSimilarity, self).__init__()
        self.weight_matrix = nn.Parameter(torch.Tensor(tensor_1_dim, tensor_2_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.activation = activation
        self.reset_parameters()
 
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        self.bias.data.fill_(0)
 
    def forward(self, tensor_1, tensor_2):
        intermediate = torch.matmul(tensor_1, self.weight_matrix)
        result = (intermediate * tensor_2).sum(dim=-1) + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result