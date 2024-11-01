from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T

import numpy as np
from torch_geometric.data import Data
import os

from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv
from typing import Union



##需要定义一下模型
class SAGE(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(SAGE, self).__init__()   ##初始化

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm: 
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)


##模型参数保持一致
mlp_parameters = {
    'lr': 0.01
    , 'num_layers': 2
    , 'hidden_channels': 128
    , 'dropout': 0.0
    , 'batchnorm': False
    , 'weight_decay': 5e-7
                  }
epochs = 200
log_steps =10 # log记录周期


para_dict = mlp_parameters
model_para = mlp_parameters.copy()
model_para.pop('lr')
model_para.pop('weight_decay')
##model = MLP(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)   不用重新创建模型了
print(f'Model sage initialized')


eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)


# 这里可以加载你的模型，加载模型映射到CPU
state_dict=torch.load('./results/model_sage.pt',map_location=torch.device('cpu'))
# 加载state_dict到模型
model=SAGE(in_channels=20, out_channels=2,**model_para)
model.load_state_dict(state_dict)

def predict(data,node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """

    # 模型预测时，测试数据已经进行了归一化处理
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    load_results_data= torch.load("./results/results_data_pred.pt", map_location=torch.device('cpu'))
    out=load_results_data['y_pred']
    out=out[node_id]
    y_pred = out.exp()  

    return y_pred

