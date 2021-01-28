#!/usr/bin/env python
#coding:utf-8
'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2020-11-02 09:24:31
LastEditors: ZhangHongYu
LastEditTime: 2020-11-02 10:10:53
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
data = {
'优先级1': [1200, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
'优先级2': [1300, 1300, 1400, 1400, 1500, 1700, 1700, 1800, 1900, 1900],
'优先级3': [1250, 1300, 1300, 1500, 1600, 1650, 1650, 1800, 1850, 1850],
"优先级4": [1350, 1350, 1400, 1400, 1550, 1550, 1600, 1700, 1800, 1950]
}
df = pd.DataFrame(data)
df.plot.box(title="4种优先级出租车收入箱线图")
plt.grid(linestyle="--", alpha=0.3)
plt.savefig("2019C/4种出租车司机收入箱线图.png")