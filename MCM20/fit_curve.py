'''
Descripttion:read the data
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-25 13:49:28
LastEditors: ZhangHongYu
LastEditTime: 2021-01-29 00:07:30
'''
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import torch
import torch.nn as nn

year = 2019
root_path = '/mnt/mydisk/LocalCode/data/OceanData/'
long_range = slice(3800, 4400)
lati_range = slice(400, 1100)
herring_plot_path = 'MCM20/data/herring_plot.csv'
mackerel_plot_path = 'MCM20/data/mackerel_plot.csv'
model_save_root = '/mnt/mydisk/LocalCode/model/MCM20/'


class Net(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        # 初始网络的内部结构
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(num_inputs, num_hiddens)
        self.linear2 = torch.nn.Linear(num_hiddens, num_outputs)
        self.relu = torch.nn.ReLU()
        # 按正态分布初始化网络参数
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.01)

    def forward(self, x):
        h = self.relu(self.linear1(x))
        out = self.linear2(h)
        return out


# 方块化热力图
class BlockMatrix(object):
    def __init__(self, long, lati, data, year):
        self.long = long
        self.lati = lati
        self.data = data
        # self.data = np.maximum(data, -20)
        self.year = year

    # 画出初始数据检测图
    def draw_data(self, mackerel_plot, herring_plot):
        mackerel_lon = mackerel_plot['long'].to_numpy()
        mackerel_lat = mackerel_plot['lati'].to_numpy()
        herring_lon = herring_plot['long'].to_numpy()
        herring_lat = herring_plot['lati'].to_numpy()
        plt.contourf(self.long, self.lati, self.data)  # 转为摄氏度
        plt.colorbar()
        plt.scatter(mackerel_lon, mackerel_lat, color='yellow')
        plt.scatter(herring_lon, herring_lat, color='black')

        #  因为鱼群是沿南北方向运动，故令lat为x, lon为y
        x1 = torch.Tensor(mackerel_lat.reshape(-1, 1))
        y1 = torch.Tensor(mackerel_lon.reshape(-1, 1))
        x2 = torch.Tensor(herring_lat.reshape(-1, 1))
        y2 = torch.Tensor(herring_lon.reshape(-1, 1))

        net1 = Net(num_inputs=1, num_hiddens=256, num_outputs=1)
        net2 = Net(num_inputs=1, num_hiddens=256, num_outputs=1)
        loss = torch.nn.MSELoss()
        optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.5)
        optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.5)

        train(net1, loss, optimizer1, x1, y1, 'mackerel')
        train(net2, loss, optimizer2, x2, y2, 'herring')

        m_state_dict1 = torch.load(model_save_root+'mackerel_plot.pt')
        net1.load_state_dict(m_state_dict1)
        m_state_dict2 = torch.load(model_save_root+'herring_plot.pt')
        net2.load_state_dict(m_state_dict2)

        y_pred1 = net1(torch.linspace(45, 60, 100).reshape(-1, 1))
        y_pred2 = net2(torch.linspace(52, 60, 50).reshape(-1, 1))

        plt.plot(
            np.squeeze(
                y_pred1.data.numpy(), 1),
            np.linspace(45, 60, 100),
            color='yellow', linestyle='--', lw=1)
        plt.plot(
            np.squeeze(
                y_pred2.data.numpy(), 1),
            np.linspace(52, 60, 50), color='black', linestyle='--', lw=1)

        plt.savefig('MCM20/data/data'+str(year)+'.png')


def train(net, loss, optimizer, x, y, fishtype):
    # 使用数据 进行正向训练，并对Variable变量进行反向梯度传播  启动100次训练
    for t in range(1000):
        # 使用全量数据 进行正向行走
        prediction = net(x)
        loss_v = loss(prediction, y)
        optimizer.zero_grad()  # 清除上一梯度
        loss_v.backward()  # 反向传播计算梯度
        optimizer.step()  # 应用梯度
    torch.save(net.state_dict(), model_save_root+fishtype+'_plot.pt')


if __name__ == '__main__':
    file_obj = nc.Dataset(root_path + 'ocean_data'+str(year) +'.nc')
    mackerel_plot = pd.read_csv(mackerel_plot_path)
    herring_plot = pd.read_csv(herring_plot_path)


    # 查看标准化后的时间
    time = file_obj.variables['time']
    time_s = str(nc.num2date(time[0], 'seconds since 1981-01-01 00:00:00'))
    year = time_s[:4]
    # for var in file_obj.variables.keys():
    #     data = file_obj.variables[var][:].data
    #     print(var, data.shape)

    long = file_obj.variables['lon'][long_range]
    lati = file_obj.variables['lat'][lati_range]
    data = (
        file_obj.variables['sea_surface_temperature']
        [0, lati_range, long_range])-274

    # 绘制北大西洋的方块热力图
    block_matrix = BlockMatrix(long, lati, data, int(year))

    block_matrix.draw_data(mackerel_plot, herring_plot)

