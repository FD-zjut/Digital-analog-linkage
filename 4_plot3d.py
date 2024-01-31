import matplotlib.pyplot as plt
import numpy as np
import pickle

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


class Cutter:
    def __init__(self,eid):
        """
        读取刀具数据
        :param eid: 刀具编号
        """
        assert 0 <= eid <= 18
        raw_data = FileIO.pickle_load("id"+str(eid))
        self.fin_result = raw_data[1][1:]
        self.target = raw_data[2][1:]
        self.sigma2 = raw_data[3][1:]
        self.CHI = raw_data[4][1:]
        self.k = raw_data[6][1:]


def plot3d(cutter_1:Cutter):
    # _ans_p_y, _ans_sigma2, _ans_real_y, _target[0], min_size
    mean = cutter_1.fin_result
    var = cutter_1.sigma2
    target = cutter_1.target
    k = cutter_1.k
    max_len = cutter_1.target[0] + 9
    begin_time = 1
    # mean, var, target, max_len, begin_time
    def gaussian(a, mean, var):
        PI = 3.14159265358979323
        a = -((a - mean) ** 2 / (2 * var))
        a = np.exp(a)
        a *= 1 / np.sqrt(2 * var * PI)
        return a
    # 获取长度
    _len = mean.shape[0]
    # 除以 renta的平方
    _var = var / (k[1:])   # 前面计算时将k除300了，这里乘回来
    # 最小磨损和最大磨损(显示用)
    L_MS = -30
    M_MS = max_len + 100
    # 设置画图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(18, 10), facecolor='white')  # 创建图片
    sub = fig.add_subplot(111, projection='3d')  # 添加子图，
    # 实际磨损量
    x1 = np.linspace(begin_time, _len + begin_time, _len)
    y1 = np.linspace(L_MS, M_MS, _len)
    z1 = np.linspace(0, 0, _len)

    for i in range(_len):
        y1[i] = target[i]
    # for i in range(_len):
    #     print(i, x1[i], y1[i], z1[i], sep='\t')
    sub.plot(x1, y1, z1, label="实际剩余寿命", color="red")

    sub.plot(x1, mean, z1, label="预测剩余寿命", color="blue")

    # 只显示一部分点
    x2 = x1.copy()
    y2 = y1.copy()
    z2 = z1.copy()
    count = 0
    _mean = mean.copy()
    for i in range(0, _len, 3):
        x2[count] = x2[i]
        y2[count] = y2[i]
        _mean[count] = _mean[i]
    x2 = x2[:count]
    y2 = y2[:count]
    z2 = z2[:count]
    _mean = _mean[:count]
    sub.scatter(x2, y2, z2, color="red")
    sub.scatter(x2, _mean, z2, color="blue")
    # 预测磨损量
    for i in range(30, _len, 30):
        # print(_len)
        x2 = np.linspace(i + begin_time, i + begin_time, _len * 10)
        dis_ = mean[i] * _var[i] * 300
        y2 = np.linspace(max(L_MS, mean[i] - dis_), min(mean[i] + dis_, M_MS), _len * 10)
        np.set_printoptions(threshold=np.inf)
        y2[0] = L_MS
        y2[-1] = M_MS
        z2 = gaussian(y2, mean[i], _var[i] * mean[i])
        print("sigma",i,_var[i] * mean[i])
        # print(z2)
        # print("var", _var[i])
        # sub.plot(x1, y1, z1, color="gray")
        if i == 0:
            sub.plot(x2, y2, z2, label="概率密度函数", color="gray")
        else:
            sub.plot(x2, y2, z2, color="gray")

    sub.set_xlabel(r"运行时间", fontsize=20, labelpad=18)
    sub.set_ylabel(r"剩余寿命", fontsize=20, labelpad=15)

    sub.set_zlabel(r"概率密度", fontsize=20, rotation=90, labelpad=15)
    # plt.yticks(size=12, weight='bold')  # 设置大小及加粗
    # plt.xticks(size=12)
    # plt.zticks(size=12)
    plt.tick_params(labelsize=18)
    plt.legend(loc="upper center", fontsize=18)  # 设置图例位置
    plt.show()

def print_data(cutter_1):
    mean = cutter_1.fin_result
    var = cutter_1.sigma2
    target = cutter_1.target
    k = cutter_1.k
    chi = cutter_1.CHI

    # 方差
    chi_2 = chi[1:].copy() - chi[0]
    chi_2 /= np.arange(1,len(chi_2)+1,1)
    print("k期望", np.mean(chi_2))
    print("k方差",np.var(chi_2))
    print("扩散系数", var[-1])
    # t = target
    # c1_features = mean

    # 皮尔逊系数
    # score=np.dot(t-np.mean(t),c1_features-np.mean(c1_features))\
    #               /np.sqrt((np.sum((t-np.mean(t))**2)*np.sum((c1_features-np.mean(c1_features))**2)))


    # print(np.mean(k))
    # print(np.var(k))

    #9.406193355538423e-06
    # print(chi)
    # print("k",np.var(k))
    # print(var)
    pass
print_data(Cutter(0))
print_data(Cutter(1))
print_data(Cutter(2))
# print_data(Cutter(1))
# print_data(Cutter(2))
# print_data(Cutter(3))
# print_data(Cutter(4))
# print_data(Cutter(5))
# plot3d(Cutter(0))
# plot3d(Cutter(1))
# plot3d(Cutter(2))
# plot3d(Cutter(3))
# plot3d(Cutter(4))
# plot3d(Cutter(5))
print("OK")