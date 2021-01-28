'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-17 20:02:03
LastEditors: ZhangHongYu
LastEditTime: 2021-01-23 22:14:23
'''
import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

data1=pd.read_csv('MCM19/ACS_10_5YR_DP02_with_ann.csv',header = 1)
data2 = pd.read_excel('MCM19/MCM_NFLIS_Data.xlsx',sheet_name=1)
id_y = pd.DataFrame({'nums':data2.groupby('FIPS_Combined')['DrugReports'].sum()})
y_X = pd.merge(data1, id_y, left_on = 'Id2', right_on = 'FIPS_Combined') 



X = pd.DataFrame()
for col_name, col in y_X.iloc[:,3:].items():
    if isinstance(col[0], str):
        X[col_name]= col
y = y_X['nums']
lr = linear_model.LinearRegression()
# lr.fit(np.array(X),np.array(y))
# joblib.dump(lr, 'MCM19/model/lr.pkl')
lr = joblib.load('MCM19/model/lr.pkl')
coef = lr.coef_

features = pd.DataFrame(columns = ['feature_name', 'weight'])
for col_name , w in zip(y_X.iloc[:,3:].columns, coef):
   features = features.append({'feature_name':col_name, 'weight': abs(w)}, ignore_index=True)

features = features.sort_values(by = 'weight', ascending = False)
features.to_csv('MCM_19/data/features.csv',index = False)

print(features)
plt.figure()
figure, ax = plt.subplots(figsize=(12, 12))
df = y_X[features.iloc[:4,0]]
sns.heatmap(df.corr(), square=True, annot=True, ax=ax)
plt.savefig('MCM_19/data/前4个特征相关性分析.png')


# for i in range(4):
#     plt.figure()
#     x = y_X[features.iloc[i,0]]
#     y = y_X['nums']
#     plt.scatter(x, y)
#     plt.savefig('第'+str(i)+'个特征的散点变化图.png')


