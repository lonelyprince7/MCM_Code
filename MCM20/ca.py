'''
Descripttion:read the data
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-25 13:49:28
LastEditors: ZhangHongYu
LastEditTime: 2021-01-28 17:36:42
'''
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import time
import random
import numpy.ma as ma
from sklearn.cluster import KMeans

year = 2019
root_path = '/mnt/mydisk/LocalCode/data/OceanData/'
long_range = slice(3800, 4400)
lati_range = slice(400, 1100)


# 方块化热力图
class BlockMatrix(object):
    def __init__(self, long, lati, data, year):
        self.long = long
        self.lati = lati
        self.year = year
        self.data = data
        self.height = data.shape[0]  # 地图x方向总高度
        self.width = data.shape[1]  # 地图y方向总宽度
        self.block_size = 20  # 每个块的大小
        self.num_x = data.shape[0]//self.block_size   # x方向块个数
        self.num_y = data.shape[1]//self.block_size  # y方向块个数
        # 用于元胞计算时的方格图，边界设置值为0的环晕
        self.map = ma.zeros((self.num_x, self.num_y), np.float)
        self.wave = 0.5
        self.alpha = 0.8
        self.belta = 1 - self.alpha
        self.setBlock()
        self.rise_coeff = 0.1

    # 方块化
    def setBlock(self):
        for x in range(self.num_x):  # 遍历x方向块
            for y in range(self.num_y):  # 遍历y方向块
                # 遍历块内的区域
                range_x = slice(x*self.block_size, (x+1)*self.block_size)
                range_y = slice(y*self.block_size, (y+1)*self.block_size)
                mean_val = np.mean(self.data[range_x, range_y])
                self.map[x, y] = mean_val
                self.data[range_x, range_y] = mean_val

    # 根据现有的温度聚类为几个区域
    def cluster(self):
        k = 5
        features = np.zeros((self.height * self.width, 1), dtype=float)
        cluster_data = self.data.copy()
        masked_lon, masked_la = [], []
        for idx_x, la in enumerate(self.lati):
            for idx_y, lon in enumerate(self.long):
                if type(self.data[idx_x, idx_y]) == np.ma.core.MaskedConstant:
                    masked_lon.append(lon)
                    masked_la.append(la)
                else:
                    if np.isnan(self.data[idx_x, idx_y]):
                        features[
                            self.width * idx_x + idx_y
                            ][0] = 0
                    else:
                        features[
                                self.width * idx_x + idx_y
                                ][0] = self.data[idx_x, idx_y]
        y_pred = KMeans(n_clusters=k, random_state=9).fit_predict(features)
        point_list = [[[], []] for i in range(k)]
        sum_list = [0.0 for i in range(k)]
        n_list = [0 for i in range(k)]
        for idx_x, la in enumerate(self.lati):
            for idx_y, lon in enumerate(self.long):
                value = y_pred[idx_x * self.width + idx_y]
                sum_list[value] += features[
                            self.width * idx_x + idx_y
                            ][0]
                n_list[value] += 1
                point_list[value][0].append(lon)
                point_list[value][1].append(la)
        mean_list = [s//n if n != 0 else 0 for s, n in zip(sum_list, n_list)]
        sorted_list = sorted(mean_list)
        colors = [
            '#FFFF00', '#FFA500', '#FF8C00', '#FF4500', '#FF0000']
        dic = {}
        for idx, temp in enumerate(sorted_list):
            dic[temp] = colors[idx]
        b_map = Basemap(
            resolution='l', area_thresh=10000, projection='cyl',
            llcrnrlon=min(self.long), urcrnrlon=max(self.long),
            llcrnrlat=min(self.lati), urcrnrlat=max(self.lati))
        for value in range(k):
            b_map.scatter(
                point_list[value][0], point_list[value][1],
                facecolor=dic[mean_list[value]])
        b_map.scatter(masked_lon, masked_la, facecolor='#FFFFFF')
        b_map.drawcoastlines(linewidth=1)
        b_map.fillcontinents(color='white', zorder=1)
        b_map.drawcountries(linewidth=1.5)
        plt.title(str(self.year)+'\'s Cluster', size=20)
        plt.savefig('MCM20/data/cluster'+str(self.year)+'.png', dpi=300)
        plt.close()

    # 元胞自动机更新一次
    def update(self):
        old = self.map.copy()
        for x in range(self.num_x):  # 遍历x方向块
            for y in range(self.num_y):  # 遍历y方向块
                if x == 0 or y == 0 or x == (
                                            self.num_x - 1
                                            ) or y == (self.num_y - 1):
                    continue
                if type(old[x, y]) == np.ma.core.MaskedConstant:
                    self.map[x, y] = old[x, y]
                    continue
                moore = [
                    [-1, 0], [0, -1], [1, 0], [0, 1],
                    [-1, -1], [1, 1], [1, -1], [-1, 1]
                ]
                moore_effective = [
                    p for p in moore if type(
                        self.data[
                            x + p[0], y + p[1]]
                            ) != np.ma.core.MaskedConstant]
                coeff = np.random.normal(
                    loc=1, scale=1, size=(len(moore_effective), 1))
                coeff /= coeff.sum()
                # 生成随机正态分布数。
                rand = np.random.normal(
                    loc=1, scale=0.05, size=(1, 1))[0]
                environ = 0.0
                for p, coe in zip(moore_effective, coeff):
                    ennviron += old[x+p[0], y+p[1]] * coe

                self.map[x, y] = self.alpha * old[x, y] + self.belta * environ + self.rise_coeff * rand

                # 温度自由抖动
                self.map[x, y] += self.wave * random.uniform(-1, 1)
        # 更新 self.data
        for x in range(self.num_x):  # 遍历x方向块
            for y in range(self.num_y):  # 遍历y方向块
                # 遍历块内的区域
                range_x = slice(x*self.block_size, (x+1)*self.block_size)
                range_y = slice(y*self.block_size, (y+1)*self.block_size)
                self.data[range_x, range_y] = self.map[x, y]
        self.year += 10

    # 画出热力图，并用用Basemap画地图
    def draw_heatmap(self):
        level_Tair = [-20, -10, 0, 5, 10, 15, 20, 25, 30, 1000]
        colors = [
            '#FFFFFF', '#AAF0FF', '#C8DC32',  '#FFBE14', '#FF780A',
            '#FF5A0A', '#F02800',  '#780A00', '#140A00']
        b_map = Basemap(
            resolution='l', area_thresh=10000, projection='cyl',
            llcrnrlon=min(self.long), urcrnrlon=max(self.long),
            llcrnrlat=min(self.lati), urcrnrlat=max(self.lati))
        # llcrnrlon=0, urcrnrlon=360, llcrnrlat=-90,urcrnrlat=90
        fig = plt.figure(figsize=(9, 6))  # plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        lon, lat = np.meshgrid(self.long, self.lati)
        x, y = b_map(lon, lat)
        cs = b_map.contourf(x, y, self.data, levels=level_Tair, colors=colors)
        # target[0,:,:]
        b_map.colorbar(cs)
        b_map.drawcoastlines(linewidth=1)
        b_map.drawcountries(linewidth=1.5)
        b_map.fillcontinents(color='white', zorder=1)
        plt.title(str(self.year)+'\'s Temperature', size=20)
        plt.savefig('MCM20/data/temp'+str(self.year)+'.png', dpi=300)
        plt.close()


if __name__ == '__main__':
    file_obj = nc.Dataset(root_path+'ocean_data'+str(year)+'.nc')

    # 查看标准化后的时间
    time = file_obj.variables['time']
    time_s = str(nc.num2date(time[0], 'seconds since 1981-01-01 00:00:00'))

    # for var in file_obj.variables.keys():
    #     data = file_obj.variables[var][:].data
    #     print(var, data.shape)

    # 截取出北大西洋的部分 2500:4500 0:2000
    long = file_obj.variables['lon'][long_range]
    lati = file_obj.variables['lat'][lati_range]
    data = file_obj.variables['sea_surface_temperature'][0, lati_range, long_range]-274

    # 绘制北爱尔兰周边海域的方块热力图
    block_matrix = BlockMatrix(long, lati, data, year)
    draw_year = 5
    for i in range(draw_year):
        block_matrix.draw_heatmap()
        block_matrix.cluster()
        block_matrix.update()
        print("iteration %d finished!" % i)
    block_matrix.draw_heatmap()
    block_matrix.cluster()