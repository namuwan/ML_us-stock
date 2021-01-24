#自作関数
import my_function_individual as mfiv
#import InvSim as IS
import Market as MK
import ChartData as CD

# data analysis
import pandas as pd
import numpy as np
import random as rnd
import math

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as dates

import sys
import time
#import matplotlib

# machine learning
#クラス名を簡潔にするときの呼び方
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Lasso
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import pickle
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=2, suppress=True)


market_file = 'data/sp500_daily_data.csv'
df_market = pd.read_csv(market_file)
market_chart = df_market.values
market_chart = np.delete(market_chart, 0, axis=1) #インデックス列を削除
market_chart = np.delete(market_chart, 0, axis=0) #インデックス列を削除
market_chart = market_chart.flatten().astype(np.float32)
log_market_chart = np.zeros(market_chart.size)
for i in range(market_chart.size):
    log_market_chart[i] = math.log10(market_chart[i]/market_chart[0])


stock_chart_file = 'data/stock_daily_data.csv'
df = pd.read_csv(stock_chart_file)
stock_chart = df.values
stock_chart = np.delete(stock_chart, 0, axis=1) #インデックス列を削除
stock_chart = np.delete(stock_chart, 0, axis=0) #インデックス行を削除
stock_chart = np.delete(stock_chart, 0, axis=1) #SPY列を削除
stock_chart = stock_chart.astype(np.float32)
print(stock_chart.shape)
print(stock_chart[0,:])
columns_list = df.columns.values.tolist()
del columns_list[:2]
print(columns_list)
print(len(columns_list))
log_stock_chart = np.zeros(stock_chart.shape)
for j in range(stock_chart.shape[1]):
    first_value = 0
    for i in range(stock_chart.shape[0]):
        if first_value==0:
            first_value = stock_chart[i][j]
        if stock_chart[i][j]!=0:
            log_stock_chart[i][j] = math.log10(stock_chart[i][j]/first_value)
#plt.plot(log_stock_chart[:,0])
#plt.show()

stock_num = stock_chart.shape[1]
return_mat = np.zeros(stock_num)
for i in range(stock_num):
    if log_stock_chart[2000][i]!=0:
        return_mat[i] = stock_chart[2600][i]/stock_chart[2300][i]
print(return_mat)
return_rank_index = np.argsort(return_mat)[::-1]
print(return_rank_index)


plt.subplot(2,1,1)
plt.plot(log_stock_chart[:,return_rank_index[0]], label='rank0')
plt.plot(log_stock_chart[:,return_rank_index[1]], label='rank1')
plt.plot(log_stock_chart[:,return_rank_index[2]], label='rank2')
plt.plot(log_stock_chart[:,return_rank_index[3]], label='rank3')
plt.legend()

plt.subplot(2,1,2)
plt.plot(log_market_chart)

plt.show()
hoge
