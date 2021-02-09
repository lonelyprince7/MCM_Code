'''
Descripttion: 采用多项式插值拟合收入曲线
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-29 18:33:41
LastEditors: ZhangHongYu
LastEditTime: 2021-01-30 18:18:47
'''
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def poly(x, a, b, c, d, e, f):
    return a*np.power(x, 5) + b*np.power(x, 4)+ c*np.power(x, 3)+ d*np.power(x, 2)+ e*x + f


def func(fish_price, boat_output, v, x):
    return fish_price * x - boat_output * 356 * 2 * (200//v)


if __name__ == '__main__':
    stomoway_data = pd.read_csv('MCM20/data/stomoway_fish.csv')
    fraseburgh_data = pd.read_csv('MCM20/data/fraseburgh_fish.csv')
    shetland_data = pd.read_csv('MCM20/data/shetland_fish.csv')
    year = stomoway_data['year'].to_numpy()

    stomoway_fish = stomoway_data['fish'].to_numpy()
    fraseburgh_fish = fraseburgh_data['fish'].to_numpy()
    shetland_fish = shetland_data['fish'].to_numpy()

    # plt.scatter(year, stomoway_fish, color='red')
    # plt.scatter(year, fraseburgh_fish, color='black')
    # plt.scatter(year, shetland_fish, color='blue')

    popt1 = [0 for i in range(6)]
    popt2 = [0 for i in range(6)]
    popt3 = [0 for i in range(6)]
    try:
        popt1, pcov1 = curve_fit(poly, year, stomoway_fish) 
    except RuntimeError:
        print("Error - curve_fit failed")
    try:
        popt3, pcov3 = curve_fit(poly, year, fraseburgh_fish) 
    except RuntimeError:
        print("Error - curve_fit failed")
    try:
        popt2, pcov2 = curve_fit(poly, year, shetland_fish) 
    except RuntimeError:
        print("Error - curve_fit failed")
    y_pred1 = np.array([poly(i, popt1[0], popt1[1], popt1[2], popt1[3], popt1[4], popt1[5]) for i in year])
    y_pred2 = np.array([poly(i, popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[5]) for i in year])
    y_pred3 = np.array([poly(i, popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[5]) for i in year])
    # plt.plot(
    #     year,
    #     func(fish_price=800, boat_output=200, v=40, x=y_pred1),
    #     color='red', linestyle='--', lw=1, label='stomoway')
    # plt.plot(
    #     year,
    #     func(fish_price=400, boat_output=200, v=40, x=y_pred1),
    #     color='black', linestyle='--', lw=1, label='fraseburgh')
    plt.plot(
        year,
        func(fish_price=660, boat_output=200, v=20, x=y_pred2),
        color='yellow', linestyle='--', lw=1, label='+10%')
    plt.plot(
        year,
        func(fish_price=630, boat_output=200, v=20, x=y_pred2),
        color='red', linestyle='--', lw=1, label='+5%')
    plt.plot(
        year,
        func(fish_price=570, boat_output=200, v=20, x=y_pred2),
        color='pink', linestyle='--', lw=1, label='-5%')
    plt.plot(
        year,
        func(fish_price=540, boat_output=200, v=20, x=y_pred2),
        color='orange', linestyle='--', lw=1, label='-10%')
    plt.plot(
        year,
        func(fish_price=600, boat_output=200, v=20, x=y_pred2),
        color='blue', linestyle='--', lw=1, label='origin')
    plt.title('profit in shetland')
    plt.legend()
    plt.savefig('MCM20/data/profit.png')
