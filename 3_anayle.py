import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import copy
import torch
import math
import networkx as nx
import torch_geometric
import torch.nn.functional as F
import pywt
import os
import re
import numpy.fft as fft
import random

from scipy.signal import savgol_filter
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from torch import nn
nomal_factor = 300
# from dataset1 import Mydataset
from torch.utils.data import DataLoader
# from earlystop import EarlyStopping
from torch_cluster import knn_graph
from torch_geometric.utils import degree,to_undirected,to_networkx
from torch_geometric.nn import GCNConv,BatchNorm
from scipy import special
from torch.utils.data import Dataset
import pickle


edge_index=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],
        [3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 0, 1, 2, 4, 5, 6, 0, 1, 2, 3, 5, 6,
         0, 1, 2, 3, 4, 6, 0, 1, 2, 3, 4, 5]])
edge_index=edge_index

class FileIO:
    @staticmethod
    def pickle_save(item, filename, folder_name='pickle/'):
        """把文件保存到指定位置并覆盖，需要路径正确（有对应文件夹），如果mode是SAVE_MODE_DICT，则保存到内存"""
        f1 = open(folder_name + filename, 'wb')
        pickle.dump(item, f1)
        f1.close()

    @staticmethod
    def pickle_load(filename, folder_name='pickle/'):
        """从指定位置的文件中读取，需要路径正确（有对应文件夹），如果mode是SAVE_MODE_DICT，则从内存中"""
        f1 = open(folder_name + filename, 'rb')
        item = pickle.load(f1)
        f1.close()
        return item

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gcn1=GCNConv(16, 8)
        # self.bn1=BatchNorm(7)
        self.gcn2=GCNConv(8, 8)
        # self.bn2=BatchNorm(7)
        # self.gcn3=GCNConv(8, 8)
        # self.fc=
        self.line=nn.Sequential(
            nn.Flatten(),
            nn.Linear(56, 24),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(24, 1),
        )
        self.miu_w = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.sigma_w=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.gamma=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.miu_w.size(0))
        self.miu_w.data.uniform_(1.5, 2)  # 随机化参数
        self.sigma_w.data.uniform_(0, stdv)  # 随机化参数
        self.gamma.data.zero_() # 随机化参数

    def forward(self, x1, edge_index):
        x1 = x1.reshape([-1,7,16])
        # print(x1.shape,"AAA")
        print("X1",x1.shape)
        print("edge_index",edge_index.shape)
        x1 = self.gcn1(x1, edge_index)
        # print(x1.shape,"AAB")
        x1 = F.relu(x1)
        # x1 = self.bn1(x1)
        x1 = self.gcn2(x1, edge_index)
        x1 = F.relu(x1)
        # x1 = self.bn2(x1)
        # x1 = self.gcn3(x2, edge_index)+x1
        x1 = self.line(x1)
        return x1

def chage_k(k):
    # chage方式A

    # chage方式B
    thersold_ = torch.tensor(0.2,dtype=torch.float32)
    k_chaged = thersold_ * (1 + torch.exp(-thersold_)) / (1 + torch.exp(-k))
    k = torch.where(k < thersold_, k_chaged, k)

    return k

def count_ans(data,data_2,dis_time,gcn,testMode_Print = False,testModeLabel=None):
    first_point =  gcn.miu_w - gcn(data, edge_index).reshape(-1)
    second_point = gcn.miu_w - gcn(data_2, edge_index).reshape(-1)

    # if (testMode_Print):
    #     print("miu_w",gcn.miu_w)
    #     print("first_point",first_point.cpu().detach().numpy())
    #     print("second_point",second_point.cpu().detach().numpy())

    k = (second_point - first_point) / dis_time
    # 修正
    k = chage_k(k)
    T_est = second_point / k * nomal_factor
    return T_est,k

def count_CHI(data,data_2,dis_time,gcn,testMode_Print = False,testModeLabel=None):
    second_point = gcn(data_2, edge_index).reshape(-1)
    return second_point





def predict_by_eid(e_id,plot3d=None):
    """
    计算一个设备全时期的寿命预测结果，以及对应的sigma
    :param e_id:设备id
    :param min_size: 进行寿命预测所需的左端时间间隔
    :param win_size: 使用窗口平均计算斜率，使用的窗口大小
    :return:
    """
    read_data = FileIO.pickle_load(str(e_id+1)+"model")
    _data = read_data[0]
    my_model = read_data[1].cpu()
    print(my_model.miu_w)
    exit()
    #生成数据
    _data = torch.tensor(_data)
    _data1 = _data[1:].clone().detach()
    _data0 = _data1.clone().detach()
    for i in range(_data0.shape[0]):
        _data0[i] = _data[0]
    _target = 314 - torch.arange(0,314)  # (200) 目前没有用到数据，但用到了它的shape
    _distime = torch.arange(1,315)
    print(_target,_distime)
    # 使用模型的结果
    _CHI = count_CHI(_data0,_data1,_distime,my_model).reshape([-1]).detach().numpy()
    _ans,_k = count_ans(_data0,_data1,_distime,my_model)
    _ans = _ans.reshape([-1]).detach().numpy()
    _k = _k.reshape([-1]).detach().numpy()
    # 使用模型的结果
    _ans_time = []
    _ans_p_y = _p_y= _ans
    _ans_real_y = _target.numpy()
    _ans_sigma2 = []
    for i in range(1,len(_p_y)):
        _ans_time.append(i)
        # 计算sigma
        dk = 0
        for j in range(i):
            dk += (_CHI[j+1] - _CHI[j])**2
        dk -= ((_CHI[i ] - _CHI[0]) ** 2)/i
        dk /= i
        _ans_sigma2.append(dk)
    print(_ans_sigma2)
    print(_k)
    FileIO.pickle_save([_ans_time, _ans_p_y, _ans_real_y, _ans_sigma2, _CHI, _data,_k/nomal_factor], "id" + str(e_id))




if __name__ == '__main__':
    for i in range(6):
        print(i)
        predict_by_eid(i)