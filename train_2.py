##将数据存到out_file里面，TEST时直接读取

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



#设置gpu设备
device = 0
device = torch.device("cuda:0")#设置device为CPU

##处理data
path='./datasets/632d74d4e2843a53167ee9a1-momodel/' #数据保存路径
save_dir='./results/' #模型保存路径
dataset_name='DGraph'
dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor()).to(device)   ##从momodel里面读入已有的数据，并将数据转为稀疏张量形式，存在device中，读到的数据为dataset

nlabels = dataset.num_classes
if dataset_name in ['DGraph']:
    nlabels = 2    #本实验中仅需预测类0和类1

data = dataset[0]   #data为data的第一个数据图
data.adj_t = data.adj_t.to_symmetric() #将有向图转化为无向图

if dataset_name in ['DGraph']:
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集

train_idx = split_idx['train']                     #train_idx为训练集
result_dir = prepare_folder(dataset_name,'SAGE')     #创建新目录保存模型（每次删除已经存在的，重新创建）


print(data)
print(data.x.shape)  #feature
print(data.y.shape)  #label


##graphsage模型
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
    

##模型参数
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


para_dict = mlp_parameters          ##para_dict为原始设置的模型参数
model_para = mlp_parameters.copy()   ##备份model的参数
model_para.pop('lr')
model_para.pop('weight_decay')      ##得到的model_para为去掉lr和weight_dacay之后的参数集合
model = SAGE(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)  ##创建一个SAGE模型model，存入GPU
print(f'Model SAGE initialized')


eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)



##train
def train(model, data, train_idx, optimizer):
     # data.y is labels of shape (N, )
    model.train()

    optimizer.zero_grad()

    out = model(data.x,data.adj_t)[train_idx]

    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

##test，评估模型在训练集和测试集上的表现
def test(model, data, split_idx, evaluator):

    with torch.no_grad():     #在 with 块中禁用梯度计算，以减少内存使用并加快计算速度，因为在评估模型时不需要反向传播。
        model.eval()          #把模型切换到eval模式（禁用dropout等只在训练时使用的layer）
        out = model(data.x, data.adj_t)     #使用data.x和data.adj_t进行forward传输，得到输出结果
        y_pred = out.exp()                  #对输出进行指数操作，得到预测值y_pred
        losses, eval_results = dict(), dict()    ##初始化losses,eval_results
        ##增加测试集test
        for key in ['train', 'valid']:
            node_id = split_idx[key]
            # out = model(data.x[node_id],data.edge_index)
            # y_pred = out.exp()  # (N,num_classes)
            #计算losses和eval_results
            losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]

            # if len(torch.unique(data.y[node_id])) > 1:
            #     eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]
            # else:
            #     eval_results[key] = float('nan')  

    return eval_results, losses, y_pred

##predict
def predict(data,node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    with torch.no_grad():
        model.eval()
        out = model(data.x,data.adj_t)[node_id]
        y_pred = out.exp()  # (N,num_classes)

    return y_pred


print(sum(p.numel() for p in model.parameters()))  #模型总参数量

model.reset_parameters()#重置参数
optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['weight_decay'])
best_valid = 0
min_valid_loss = 1e8

results_data_pred={}    ##创建字典，保存后续得到的测试数据

for epoch in range(1,epochs + 1):
    loss = train(model, data, train_idx, optimizer)
    eval_results, losses, out = test(model, data, split_idx, evaluator)

    train_eval, valid_eval = eval_results['train'], eval_results['valid']
    # test_eval=eval_results['test']    #得到test的eval_results
    train_loss, valid_loss = losses['train'], losses['valid']
    # test_loss=losses['test']     #得到test的losses

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), save_dir+'/model_sage.pt') #将表现最好的模型保存
        ##将最佳结果存入results_data_pred={}
        results_data_pred = {
            'train_eval': train_eval,
            'valid_eval': valid_eval,
            # 'test_eval': test_eval, 
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            # 'test_loss': test_loss,
            'y_pred': out  # 如果需要保存 y_pred
        }


    if epoch % log_steps == 0:
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_eval:.3f}, ' # 我们将AUC值乘上100，使其在0-100的区间内
              f'Valid: {100 * valid_eval:.3f}')
             # f'Test: {100 * test_eval:.3f}'#

if results_data_pred:  # 确保字典非空
    torch.save(results_data_pred, save_dir+'results_data_pred.pt')

model.load_state_dict(torch.load(save_dir+'/model_sage.pt')) #载入验证集上表现最好的模型



dic={0:"正常用户",1:"欺诈用户"}
node_idx = 0
y_pred = predict(data, node_idx)
print(y_pred)
print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')

node_idx = 1
y_pred = predict(data, node_idx)
print(y_pred)
print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')
