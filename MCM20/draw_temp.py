'''
Descripttion:read the data
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-25 13:49:28
LastEditors: ZhangHongYu
LastEditTime: 2021-01-28 17:44:17
'''
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import time
import random
from sklearn.cluster import KMeans

origin = 2001
root_path = '/mnt/mydisk/LocalCode/data/OceanData/'
year_range = 20
long_range = slice(3800, 4400)
lati_range = slice(400, 1100)


# 方块化热力图
class BlockMatrix(object):
    def __init__(self, long, lati, data, year):
        self.long = long
        self.lati = lati
        self.data = data
        self.year = year
        self.height = data.shape[0]  # 地图x方向总高度
        self.width = data.shape[1]  # 地图y方向总宽度

    # 方块化
    def getAvg(self):
        return data.mean()


if __name__ == '__main__':

    x, y, ticks = [], [], []
    for i in range(year_range):
        file_obj = nc.Dataset(
            root_path + 'ocean_data'+str(origin + i)+'.nc')

        # for var in file_obj.variables.keys():
        #     data = file_obj.variables[var][:].data
        #     print(var, data.shape)
        #  查看标准化后的时间

        time = file_obj.variables['time']
        time_s = str(nc.num2date(time[0], 'seconds since 1981-01-01 00:00:00'))
        print(time_s)

        # 截取出北爱尔兰周边海域部分部分
        long = file_obj.variables['lon'][long_range]
        lati = file_obj.variables['lat'][lati_range]
        data = (
            file_obj.variables['sea_surface_temperature']
            [0, lati_range, long_range])-274

        # 绘制北大西洋的方块热力图
        block_matrix = BlockMatrix(long, lati, data, origin + i)
        y.append(block_matrix.getAvg())
        ticks.append(str(origin + i))

    plt.figure(figsize=(14, 7))

    plt.scatter(
        np.arange(origin, origin + year_range).astype(
            dtype=np.str), y, color='r')
    plt.plot(
        np.arange(origin, origin + year_range).astype(dtype=np.str),
        y, color='#FFA500', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Average Temerature')
    plt.title('Historical annual mean temperature of the North Atlantic')
    plt.savefig('MCM20/data/average_temp'+str(origin)+'to'+str(origin + year_range -1 ) + '.png')
    # draw_year = 5
    # block_matrix.cluster()
    # for i in range(draw_year):
    #     block_matrix.draw()
    #     block_matrix.update()
    #     block_matrix.cluster()
    #     print("iteration %d finished!" % i)
    # block_matrix.draw()
