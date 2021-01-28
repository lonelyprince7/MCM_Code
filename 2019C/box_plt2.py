#!/usr/bin/env python
#coding:utf-8
'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2020-11-02 09:24:31
LastEditors: ZhangHongYu
LastEditTime: 2020-11-02 10:10:00
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
data = {
'划分优先级前的收入': [900, 1100, 1300, 1300, 1500, 1600, 1900, 1900, 1900, 2300],
'划分优先级后的收入': [1200, 1300, 1300, 1400, 1500, 1700, 1700, 1800, 2000, 2000],
}
df = pd.DataFrame(data)
df.plot.box(title="划分优先级前后出租车收入箱线图")
plt.grid(linestyle="--", alpha=0.3)
plt.savefig("2019C/划分优先级前后出租车司机收入箱线图.png")