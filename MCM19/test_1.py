'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2020-10-28 02:49:12
LastEditors: ZhangHongYu
LastEditTime: 2021-01-23 22:02:41
'''
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
drug_data = pd.read_excel('MCM19/MCM_NFLIS_Data.xlsx',sheet_name=1)


# sum_for_each_plot = pd.DataFrame({'FIPS_State': np.unique(drug_data['FIPS_State']),
# 'DrugReports':drug_data.groupby('FIPS_State')['DrugReports'].sum()})

#各州药物总量随年份变化
# sum_year_stat = pd.DataFrame({
# '21-DrugReports':drug_data[drug_data['FIPS_State'] == 21].groupby('YYYY')['DrugReports'].sum(),
# '39-DrugReports':drug_data[drug_data['FIPS_State'] == 39].groupby('YYYY')['DrugReports'].sum(),
# '42-DrugReports':drug_data[drug_data['FIPS_State'] == 42].groupby('YYYY')['DrugReports'].sum(),
# '51-DrugReports':drug_data[drug_data['FIPS_State'] == 51].groupby('YYYY')['DrugReports'].sum(),
# '54-DrugReports':drug_data[drug_data['FIPS_State'] == 54].groupby('YYYY')['DrugReports'].sum()
# })

# sum_year_stat.plot(kind = 'bar')


#各药物总量随年份变化
# sum_year_stat = pd.DataFrame({
# 'Heroin':drug_data[drug_data['SubstanceName'] == 'Heroin'].groupby('YYYY')['DrugReports'].sum(),
# 'Morphine':drug_data[drug_data['SubstanceName'] == 'Morphine'].groupby('YYYY')['DrugReports'].sum(),
# 'Methadone':drug_data[drug_data['SubstanceName'] == 'Methadone'].groupby('YYYY')['DrugReports'].sum(),
# 'Hydromorphone':drug_data[drug_data['SubstanceName'] == 'Hydromorphone'].groupby('YYYY')['DrugReports'].sum(),
# 'Oxycodone':drug_data[drug_data['SubstanceName'] == 'Oxycodone'].groupby('YYYY')['DrugReports'].sum(),
# 'Oxymorphone':drug_data[drug_data['SubstanceName'] == 'Oxymorphone'].groupby('YYYY')['DrugReports'].sum(),
# })


# 各州海洛因总量随年份变化
sum_year_stat = pd.DataFrame({ 
'21-Heroin':drug_data[(drug_data['FIPS_State'] == 21)&(drug_data['SubstanceName'] == 'Heroin')].groupby('YYYY')['DrugReports'].sum(),
'39-Heroin':drug_data[(drug_data['FIPS_State'] == 39)&(drug_data['SubstanceName'] == 'Heroin')].groupby('YYYY')['DrugReports'].sum(),
'42-Heroin':drug_data[(drug_data['FIPS_State'] == 42)&(drug_data['SubstanceName'] == 'Heroin')].groupby('YYYY')['DrugReports'].sum(),
'51-Heroin':drug_data[(drug_data['FIPS_State'] == 51)&(drug_data['SubstanceName'] == 'Heroin')].groupby('YYYY')['DrugReports'].sum(),
'54-Heroin':drug_data[(drug_data['FIPS_State'] == 54)&(drug_data['SubstanceName'] == 'Heroin')].groupby('YYYY')['DrugReports'].sum()
})


sum_year_stat.plot(kind = 'bar')
plt.show()


# #39号州各县海洛因分布
# heroin_county = pd.DataFrame( 
#     drug_data[(drug_data['FIPS_State'] == 39)&(drug_data['SubstanceName'] == 'Heroin')].groupby('FIPS_County')['DrugReports'].sum()
# )
# print(heroin_county.shape)
# heroin_county.plot(kind = 'pie',y='DrugReports')
# plt.show()



