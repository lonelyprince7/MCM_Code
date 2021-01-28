'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2020-10-28 02:49:12
LastEditors: ZhangHongYu
LastEditTime: 2020-11-01 21:35:32
'''
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data=pd.read_csv('2019C/data.csv')
data=np.array(data[['longitude','latitude']])
data=data.astype(float)

dis=[ math.hypot(coord[0],coord[1]) for coord in data]
dis=np.array(dis).reshape(-1,1)
y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(dis)

x0=[ coord[0] for l,coord in zip(y_pred,data) if l ==0]
y0=[ coord[1] for l,coord in zip(y_pred,data) if l ==0]
x1=[ coord[0] for l,coord in zip(y_pred,data) if l ==1]
y1=[ coord[1] for l,coord in zip(y_pred,data) if l ==1]
x2=[ coord[0] for l,coord in zip(y_pred,data) if l ==2]
y2=[ coord[1] for l,coord in zip(y_pred,data) if l ==2]
x3=[ coord[0] for l,coord in zip(y_pred,data) if l ==3]
y3=[ coord[1] for l,coord in zip(y_pred,data) if l ==3]

print(x0)
plt.figure()
plt.scatter(x0,y0,color='r')
plt.scatter(x1,y1,color='b')
plt.scatter(x2,y2,color='y')
plt.scatter(x3,y3,color='g')

plt.savefig('./2019C/出租车终点空间分布.png')