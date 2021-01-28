'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2020-10-28 02:49:12
LastEditors: ZhangHongYu
LastEditTime: 2021-01-23 22:02:53
'''
import decimal
import pandas as pd
import numpy as np
import math
import numpy 
import random
import matplotlib.pyplot as plt
class Graph(object):
    def __init__(self, n_node):
        #初始化图的邻接矩阵
        self.adj_matrix = np.zeros((n_node, n_node),dtype=float)
        self.prob = np.zeros((n_node, n_node),dtype=float)
        #初始化图的节点
        self.vertex = [ 0 for j in range(n_node)]
        self.n_node = n_node
        self.u = 0.25 #感染者痊愈的概率
    #定义sigmoid函数归一化
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))
    #计算边的权值矩阵
    def set_adj_matrix(self):
        for i in range(self.adj_matrix.shape[0]):
            for j in range(self.adj_matrix.shape[1]): 
                    self.adj_matrix[i][j] = self.sigmoid(abs(self.vertex[i] - self.vertex[j]))
    #计算感染概率矩阵
    def set_prob(self):
        self.prob = np.zeros((self.n_node, self.n_node) , dtype = float)
        for i in range(self.adj_matrix.shape[0]):
            tot_sum = 0
            #若 self.vertex[j] <= self.vertex[i]，则 prob[i][j] = 0
            for j in range(self.adj_matrix.shape[1]): 
                if self.vertex[j] > self.vertex[i]:
                    tot_sum += (1 - self.adj_matrix[i][j])
            for j in range(self.adj_matrix.shape[1]):
                if self.vertex[j] > self.vertex[i]:
                    if tot_sum < 1e-20:
                        self.prob[i][j] = 0
                    else :
                        self.prob[i][j] = ( 1 - self.adj_matrix[i][j] ) / tot_sum   
    def show_vertex(self):
        df = pd.DataFrame({'heroin':self.vertex},index=[i for i in range(self.n_node)])
        df.plot(kind='bar')
        plt.xticks([])  #去掉横坐标值
        plt.show()
    def show_adj_matrix(self):
        self.set_adj_matrix()
        plt.matshow(self.adj_matrix, cmap=plt.cm.gray)
        plt.show()
    def spread(self):
        old = self.vertex.copy()
        new = self.vertex.copy()
        self.set_adj_matrix() #更新权值矩阵
        self.set_prob() #更新感染概率矩阵
        # max_v = -1
        # max_idx = -1
        # for i in range(self.n_node):
        #    if self.vertex[i] > max_v :
        #        max_v = self.vertex[i]
        #        max_idx = i
        for i in range(self.n_node):
            # if i % 2 == 0 :
            #     continue
            v = 0
            for j in range(self.n_node):
                # if j % 2 == 0 :
                #     continue
                #每个节点 i 以 adj_matrix[i][j] 的概率被 j 感染
                diff = old[j] - old[i]
                #被周围感染
                v += 10 * self.prob[i][j] * diff
            new[i] += v
            print(v)
            #节点自动痊愈概率
            r_recover = random.random()
            if r_recover < self.u:
                new[i] = max(0, new[i] - v)
        self.vertex = new

#建立县的FIPS序号->图的节点编号的映射
def hash_map(drug_data, id_to_idx, year, cnt):
    for node_id in list(drug_data[(drug_data['SubstanceName'] == 'Heroin') & (drug_data['YYYY'] == year)]['FIPS_Combined']):
        if not (node_id in id_to_idx.keys()):
            id_to_idx[node_id]=cnt
            cnt += 1
    return cnt

#初始化每年数据所对应的图的节点
def init_vertex(drug_data, id_to_idx, vertex, year):
    ids = list(drug_data[(drug_data['SubstanceName'] == 'Heroin') & (drug_data['YYYY'] == year)]['FIPS_Combined'])
    nums = list(drug_data[(drug_data['SubstanceName'] == 'Heroin') & (drug_data['YYYY'] == year)]['DrugReports'])
    for id, num in zip(ids,nums):
        vertex[id_to_idx[id]]=num

#计算年与年之间节点之间的差值
def comput_delta(i, delta, vertex1, vertex2):
    delta[i] = [ v2 - v1 for v1, v2 in zip(vertex1, vertex2)]
       
#绘制直方图
def draw_bar(i, delta, cnt):
    print(delta[i])
    df = pd.DataFrame({'heroin':delta[i]},index=[i for i in range(cnt)])
    df.plot(kind='bar')

if __name__ == "__main__":
    drug_data = pd.read_excel('MCM19/MCM_NFLIS_Data.xlsx',sheet_name=1)

    #建立县的FIPS序号->图的节点编号的映射，因为每年上报的县的数量都不一样，这里凡是出现过一次的县就为其建立映射
    id_to_idx ={}  
    cnt=0  #cnt表示所有年份总共上报过的县的数量
    years=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    year = len(years)
    for y in years:
        cnt = hash_map(drug_data, id_to_idx, y, cnt)

    #初始化每年的图
    Graphs=[]
    for i in range(year):
        Graphs.append(Graph(cnt))


    #初始化图的节点
    for i, y in enumerate(years):
        init_vertex(drug_data, id_to_idx, Graphs[i].vertex, y)


    #记录各县年与年之间上报数量的3次差值，用于确定spread函数用
    # delta = [[] for i in range(year-1)]
    # for i in range(len(years)-1):
    #     comput_delta(i, delta, Graphs[i].vertex,Graphs[i+1].vertex)
    #     draw_bar(i, delta, cnt)
    
    #绘制每年的差值图像
    # plt.show()


    #打印每张图的邻接矩阵
    # for i in range(year-1):
    #     Graphs[i].show_adj_matrix()

    #2010的图传播一年，与2011年的图比较看是否拟合
    # Graphs[0].show_vertex()
    # Graphs[0].spread()
    # Graphs[0].spread()
    # Graphs[0].spread()
    # Graphs[0].show_vertex()
    Graphs[7].show_vertex()
    Graphs[7].spread()
    Graphs[7].spread()
    Graphs[7].show_vertex()


