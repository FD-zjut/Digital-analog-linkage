import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import copy
import torch
import math
import networkx as nx
# import torch_geometric
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
# from dataset1 import Mydataset
from torch.utils.data import DataLoader
# from earlystop import EarlyStopping
from torch_cluster import knn_graph
from torch_geometric.utils import degree,to_undirected,to_networkx
from torch_geometric.nn import GCNConv,BatchNorm
from scipy import special
from torch.utils.data import Dataset
nomal_factor = 300
import pickle

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TRAIN_TOOLS = [4,6]
TEST_TOOLS = [1]

class Mydataset(Dataset):
    def __init__(self, data,second_data,dis_time,label):
        self.data = data
        self.second_data = second_data
        self.dis_time = dis_time
        self.label = label
            
    # 根据索引获取data和label
    def __getitem__(self, index):
        return self.data[index], self.second_data[index], self.dis_time[index], self.label[index]  # 以元组的形式返回

    # 获取数据集的大小
    def __len__(self):
        return len(self.data)

## 考虑了阈值V的不确定性，theta的不确定性和扩散系数的不确定性
seed=5#随机种子

random.seed(seed)
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
device = torch.device("cuda")    



edge_index=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],
        [3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 0, 1, 2, 4, 5, 6, 0, 1, 2, 3, 5, 6,
         0, 1, 2, 3, 4, 6, 0, 1, 2, 3, 4, 5]])
edge_index=edge_index.to(device)


## 读取数据
raw_data_set = []
for c_id in ["c1","c4","c6"]:
    for i_id in range(10):
        delete_index = np.array([1, 3, 5, 8, 10, 11, 12, 13, 14, 16, 17, 20])
        temp = np.load(c_id+'_features'+'.npy').astype('float32')
        temp = np.delete(temp, delete_index, 2)
        # 归一化
        for i in range(7):
            for j in range(16):
                g1 = np.expand_dims(temp[:, i, j], 1)
                temp[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g1), 1)  # 归一化
                temp[:, i, j] *= (1 + np.random.randn(315) * 0.1)
                # print(np.random.randn(315))
        raw_data_set.append(temp)
first_point_set = []
second_point_set = []
dis_time_point_set = []

FIRST_POINT_RANGE = 314
# for item in raw_data_set:
#     second_point_set.append(item[1:FIRST_POINT_RANGE])

for item  in raw_data_set:
    temp_sec = torch.tensor(item[1:FIRST_POINT_RANGE]).to(device)
    temp_f = torch.zeros_like(temp_sec)
    temp_time = torch.zeros(temp_f.shape[0])
    for j in range(1, FIRST_POINT_RANGE):
        now_id = j - 1
        temp_f[now_id, :, :] = torch.tensor(item[0]).to(device)
        temp_time[now_id] = j

    first_point_set.append(temp_f)
    second_point_set.append(temp_sec)
    dis_time_point_set.append(temp_time)
ALL_TOOLS = [i for i in range(30)]
TEST_TOOLS = [3,6,13,16,23,26]
SUBED_TOOL = list(set(ALL_TOOLS) - set(TEST_TOOLS))

# 通过合并的方式，生成训练集和测试集
def make_data(TEST_TOOLS,first_point_set,second_point_set,dis_time_point_set):
    data = None
    data_2 = None
    dis_time = None
    for i in range(len(raw_data_set)):
        if i not in TEST_TOOLS:
            if data is None:
                data = first_point_set[i]
                data_2 = second_point_set[i]
                dis_time = dis_time_point_set[i]
            else:
                data = torch.cat((data, first_point_set[i]), 0)
                data_2 = torch.cat((data_2, second_point_set[i]), 0)
                dis_time = torch.cat((dis_time, dis_time_point_set[i]), 0)
    return data,data_2,dis_time
data,data_2,dis_time = make_data(TEST_TOOLS,first_point_set,second_point_set,dis_time_point_set)
data_2 = data_2.to(device)
dis_time = dis_time.to(device)
print(data_2)

label=np.expand_dims(np.flip(np.arange(1,FIRST_POINT_RANGE,1)),1).copy()
label=np.array(label,dtype='float32')
label=torch.tensor(label).repeat(24,1).to(device)

# print(label.shape)
# print(data.shape)
# print(data_2.shape)
# for i in range(20):
#     print(i)
#     print(data[i][0][0])
#     print(label[i][0])
#     print("+"*20)
# exit()
# label=torch.cat((label,label,label),0)


## 构建训练数据集
train_data = Mydataset(data,data_2,dis_time,label)
train_size=60
train_data = DataLoader(train_data, batch_size=train_size,shuffle=True)

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

# 创建网络模型
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
    
gcn = GCN()
gcn=gcn.to(device)

# 损失函数
loss_fn = nn.MSELoss()
loss_fn=loss_fn.to(device)

# 优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(gcn.parameters(),lr=learning_rate,weight_decay=0.1)


# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 2500

loss_train =[]
loss_test  =[]
accuracy_test=[]

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
    return T_est


def count_Test(data, data_2, dis_time, gcn, testMode_Print = False, testModeLabel=None):
    second_point = gcn(data_2, edge_index).reshape(-1)
    first_point = gcn(data, edge_index).reshape(-1)
    k = (second_point - first_point) / dis_time
    return second_point, first_point,k


a1=0
loss_list_c1 = []

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))
    
    # 训练步骤开始
    gcn.train()
    total_train_loss=0
    total_train_accuracy=0
    
    for x in train_data:
        data,data_2,dis_time,label=x

        label = label.reshape(-1)

        T_est=count_ans(data,data_2,dis_time,gcn,testModeLabel=label)
        # print(first_point.shape,second_point.shape,k.shape,T_est.shape,"@DD")
        CHI = gcn(data_2, edge_index).reshape(-1)
        ##计算准确度
        error_train=label-T_est
        if total_train_step % 200 == 150:
            np.set_printoptions(threshold=np.inf, precision=1, linewidth=np.inf)
        zero = torch.zeros_like(error_train)
        one = torch.ones_like(error_train)
        error_train = torch.where(error_train < -10, zero, error_train)
        error_train = torch.where(error_train > 13, zero, error_train)
        train_accuracy = torch.count_nonzero(error_train)
        total_train_accuracy = total_train_accuracy + train_accuracy.item()

        loss1 = loss_fn(T_est, label)
        loss = loss_fn(T_est, label) \
                    + torch.sum(torch.where(CHI>0,torch.zeros_like(CHI).cuda(),torch.pow(CHI,2)))
            # +(1/(100*torch.maximum(torch.tensor(0).to(device),torch.mean(gcn.miu_w-head))+0.0001)) 


        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss1.item()))

    print(train_data.dataset.data.shape)
    total_train_accuracy = total_train_accuracy/train_data.dataset.data.shape[0]*100
    if (total_train_accuracy>a1):
        a1=total_train_accuracy
    print("训练集准确率 {}%".format(total_train_accuracy))


    
    if i % 2 == 0:
        IF_PRINT = False
        if i %16 ==0:
            IF_PRINT = True
        c1 = raw_data_set[3]
        c4 = raw_data_set[16]
        c6 = raw_data_set[23]
        c1_2 = raw_data_set[6]
        c4_2 = raw_data_set[13]
        c6_2 = raw_data_set[26]
        # print("SDSDD",c4_2-c1_2)
        c=[c1,c4,c6,c1_2,c4_2,c6_2]
        Te = np.zeros([312,len(c)])
        CHIS = np.zeros([312, len(c)])
        CHIS_Compare = np.zeros([312, len(c)])
        cut_id=0
        dao_rul=np.flip(np.arange(1,316,1)).copy()
        dao_rul=torch.unsqueeze(torch.tensor(dao_rul), 1).to(device)

        for d in c:
            dao=torch.tensor(d[0,:,:]).repeat(312,1,1).to(device)
            dao_end=torch.tensor(d[1:313,:,:]).to(device)
            dis_time = torch.arange(1,313).to(device)

            T = count_ans(dao, dao_end, dis_time, gcn,testMode_Print=True)
            CHI,first_CHI,k_ = count_Test(dao, dao_end, dis_time, gcn, testMode_Print=True)
            # 取最后一个点
            k_ = k_ *0 + k_[-10]

            # print(first_CHI[0].unsqueeze(0).shape)
            # print(CHI.shape)
            CHI_k = torch.cat([first_CHI[0].unsqueeze(0),CHI]) # 将首点考虑在内
            dis_CHI = CHI_k[1:] -  CHI_k[:-1]
            dis_CHI = torch.pow(dis_CHI,2)
            sum_dis_CHI = torch.zeros_like(dis_CHI)
            now = 0
            for i in range(1,len(sum_dis_CHI) + 1):
                now += dis_CHI[i-1]
                sum_dis_CHI[i-1] = now


            dis_time = torch.arange(1,313).to(device)
            sigma = (sum_dis_CHI - torch.pow(CHI_k[1:] -  first_CHI,2)/dis_time)/dis_time
            print("sigma", sigma)

            sigma = torch.sqrt(sigma)
            CHIS_Compare[:,cut_id] = (first_CHI + dis_time * k_ + sigma * torch.randn(sigma.shape).cuda()).cpu().detach().numpy()



            Te[:,cut_id]=T[0:312].cpu().detach().numpy()
            CHIS[:,cut_id] = CHI[0:312].cpu().detach().numpy()
            cut_id=cut_id+1
            save_data = [d,gcn]
            FileIO.pickle_save(save_data,str(cut_id)+"model")
        dao_rua=dao_rul[0:312].cpu().detach().numpy()

        if IF_PRINT:
            # plt.figure(1)
            # plt.title('刀的寿命曲线')
            # plt.subplot(321)
            # plt.plot(Te[:,0],color='red')
            # # plt.plot(CHIS[:,0])
            # plt.plot(dao_rua )
            # # plt.ayvline(y=3, color='r', linestyle='--')
            # plt.xlabel('走刀数')
            # plt.ylabel('寿命')
            # # print("c1_dis",(Te[:,0].reshape(-1) - (dao_rua ).reshape(-1)))
            #
            # plt.subplot(323)
            # plt.plot(Te[:,1],color='red')
            # # plt.plot(CHIS[:,1])
            # plt.plot(dao_rua )
            # np.set_printoptions(threshold=np.inf, precision=1, linewidth=np.inf)
            # # print(Te[:,0] - dao_rua.reshape(-1))
            # # print(Te[:,1] - dao_rua.reshape(-1))
            #
            # plt.xlabel('走刀数')
            # plt.ylabel('寿命')
            # print("c4_dis",(Te[:,1].reshape(-1) - (dao_rua ).reshape(-1)))

            # plt.subplot(325)
            # plt.plot(Te[:,2],color='red')
            # # plt.plot(CHIS[:,2])
            # plt.plot(dao_rua )
            # plt.xlabel('走刀数')
            # plt.ylabel('寿命')
            # print("c6_dis",(Te[:,2].reshape(-1) - (dao_rua ).reshape(-1)))

            # plt.subplot(322)
            # plt.plot(CHIS[:,0])
            # plt.axhline(y=gcn.miu_w.detach().item(), color='r', linestyle='--')
            # plt.xlabel('走刀数')
            # plt.ylabel('CHI')
            # plt.subplot(324)
            # plt.plot(CHIS[:,1])
            # plt.axhline(y=gcn.miu_w.detach().item(), color='r', linestyle='--')
            # plt.xlabel('走刀数')
            # plt.ylabel('CHI')
            # plt.subplot(326)
            # plt.plot(CHIS[:,2])
            # plt.axhline(y=gcn.miu_w.detach().item(), color='r', linestyle='--')
            # plt.xlabel('走刀数')
            # plt.ylabel('CHI')
            # plt.show()

            plt.subplot(311)
            plt.plot(CHIS[:,0],color='red',label="网络计算")
            plt.plot(CHIS_Compare[:, 0],color='blue',label="公式计算")
            plt.xlabel('走刀数')
            plt.ylabel('CHI')
            plt.legend()
            plt.subplot(312)
            plt.plot(CHIS[:,1],color='red',label="网络计算")
            plt.plot(CHIS_Compare[:, 1],color='blue',label="公式计算")
            plt.xlabel('走刀数')
            plt.ylabel('CHI')
            plt.legend()
            plt.subplot(313)
            plt.plot(CHIS[:,2],color='red',label="网络计算")
            plt.plot(CHIS_Compare[:, 2],color='blue',label="公式计算")
            plt.xlabel('走刀数')
            plt.ylabel('CHI')
            plt.legend()
            plt.show()

        if 1 in TEST_TOOLS:
            TEST_LINE = Te[:, 0]
        elif 4 in TEST_TOOLS:
            TEST_LINE = Te[:, 1]
        else:
            TEST_LINE = Te[:, 2]
        # TEST_LINE = Te[:, 0]
        MSE_Loss = 0
        Score = 0
        for j in range(len(c)):
            TEST_LINE = Te[:, j]
            print(np.arange(TEST_LINE.shape[0] - 1, -1, -1))
            print(TEST_LINE)
            ERR_LINE = np.arange(TEST_LINE.shape[0] - 1, -1, -1)  - TEST_LINE
            # print("MSE", np.arange(0,TEST_LINE.shape[0],1))
            MSE_Loss += np.mean(np.power(ERR_LINE, 2))

            loss_list_c1.append(MSE_Loss.item())

            def countScore(x):
                zeros = np.zeros_like(x)
                # print(np.where(x > 0,x,zeros)/13)
                # print(np.where(x < 0,x,zeros) / 10)
                x1 = np.exp(np.where(x > 0,x,zeros)/13)-1
                x2 = np.exp(np.where(x < 0,-x,zeros) / 10)-1
                # print("x1",x1)
                # print("x2",x2)
                Score = (np.sum(x1) + np.sum(x2))/x.shape[0]

                return Score
            Score += countScore(ERR_LINE)
        print("MSE", MSE_Loss / len(c))
        print("SCORE", countScore(ERR_LINE) / len(c))
        print("loss_list_c1",loss_list_c1)
    print("w", gcn.miu_w)
