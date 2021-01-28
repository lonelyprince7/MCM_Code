'''
Descripttion:read the data
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-25 13:49:28
LastEditors: ZhangHongYu
LastEditTime: 2021-01-28 21:49:13
'''
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import time
import random
from sklearn.cluster import KMeans
import pandas as pd

year = 2019
root_path = '/mnt/mydisk/LocalCode/data/OceanData/'
long_range = slice(3800, 4400)
lati_range = slice(400, 1100)
herring_plot_path = 'MCM20/data/Herring_plot.csv'
mackerel_plot_path = 'MCM20/data/Makerel_plot.csv'


# 方块化热力图
class BlockMatrix(object):
    def __init__(self, long, lati, data, year):
        self.long = long
        self.lati = lati
        self.data = data
        # self.data = np.maximum(data, -20)
        self.year = year
        self.height = data.shape[0]  # 地图x方向总高度
        self.width = data.shape[1]  # 地图y方向总宽度
        self.rise_coeff = 0.1

    # 画出初始数据检测图
    def draw_data(self, mackerel_plot):
        mackerel_lon = mackerel_plot['lon']
        mackerel_lat = mackerel_plot['lat']
        plt.scatter(mackerel_lon, mackerel_lat, color='r')
        plt.contourf(self.long, self.lati, self.data)  # 转为摄氏度
        plt.colorbar()
        plt.savefig('MCM20/data/data'+str(year)+'.png')


if __name__ == '__main__':
    file_obj = nc.Dataset(root_path + 'ocean_data'+str(year) +'.nc')
    mackerel_plot = pd.read_csv(mackerel_plot_path)


    # 查看标准化后的时间
    time = file_obj.variables['time']
    time_s = str(nc.num2date(time[0], 'seconds since 1981-01-01 00:00:00'))
    year = time_s[:4]
    # for var in file_obj.variables.keys():
    #     data = file_obj.variables[var][:].data
    #     print(var, data.shape)

    # 截取出北大西洋的部分 2500:4500 0:2000


    long = file_obj.variables['lon'][long_range]
    lati = file_obj.variables['lat'][lati_range]
    data = (
        file_obj.variables['sea_surface_temperature']
        [0, lati_range, long_range])-274

    # 绘制北大西洋的方块热力图
    block_matrix = BlockMatrix(long, lati, data, int(year))

    block_matrix.draw_data(mackerel_plot)
    # draw_year = 5
    # block_matrix.cluster()
    # for i in range(draw_year):
    #     block_matrix.draw_heatmap()
    #     block_matrix.update()
    #     block_matrix.cluster()
    #     print("iteration %d finished!" % i)
    # block_matrix.draw()
