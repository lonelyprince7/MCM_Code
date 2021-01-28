'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2020-10-28 02:49:12
LastEditors: ZhangHongYu
LastEditTime: 2021-01-23 22:04:31
'''
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
drug_data = pd.read_excel('MCM19/MCM_NFLIS_Data.xlsx',sheet_name=1)
print(drug_data.loc[(drug_data['COUNTY'] == 'ADAMS')])
data1=pd.read_csv('MCM19/MCM19_5YR_DP02_with_ann.csv',header = 1)
print(data1.shape)
data2=pd.read_csv('MCM19/MCM19_5YR_DP02_metadata.csv',header = 1)
# print(data2.iloc[0:,:])
print(data1['Geography'])
# print(data2.columns)