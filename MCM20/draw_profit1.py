'''
Descripttion: 采用多项式插值拟合收入曲线
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-29 18:33:41
LastEditors: ZhangHongYu
LastEditTime: 2021-01-30 01:03:13
'''
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def func(x, a, b, c, d, e, f):
    return a*np.power(x, 5) + b*np.power(x, 4)+ c*np.power(x, 3)+ d*np.power(x, 2)+ e*x + f


if __name__ == '__main__':
    stomoway_data = pd.read_csv('MCM20/data/stomoway_profit.csv')
    shetland_data = pd.read_csv('MCM20/data/shetland_profit.csv')
    year = stomoway_data['year'].to_numpy()

    stomoway_profit = stomoway_data['profit'].to_numpy()
    shetland_profit = shetland_data['profit'].to_numpy()

    plt.scatter(year, stomoway_profit, color='red')
    plt.scatter(year, shetland_profit, color='blue')
    popt1 = [0 for i in range(6)]
    popt2 = [0 for i in range(6)]
    try:
        popt1, pcov1 = curve_fit(func, year, stomoway_profit) 
    except RuntimeError:
        print("Error - curve_fit failed")
    try:
        popt2, pcov2 = curve_fit(func, year, shetland_profit) 
    except RuntimeError:
        print("Error - curve_fit failed")
    y_pred1 = [func(i, popt1[0], popt1[1], popt1[2], popt1[3], popt1[4], popt1[5]) for i in year]    
    y_pred2 = [func(i, popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[5]) for i in year]  

    plt.plot(
        year,
        y_pred1,
        color='red', linestyle='--', lw=1)
    plt.plot(
        year,
        y_pred2,
        color='blue', linestyle='--', lw=1)
    plt.show()
    plt.savefig('MCM20/data/profit.png')
