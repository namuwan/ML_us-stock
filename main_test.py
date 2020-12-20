#coding:utf-8
########################
#choice asset
########################
#%matplotlib inline

#自作関数
import my_function_individual as mfiv
#import InvSim as IS
import Market as MK
import ChartData as CD

# data analysis
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as dates

import sys
import time
#import matplotlib

# machine learning
#クラス名を簡潔にするときの呼び方
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import pickle


filename = 'data/stock_daily_data.csv' #日次
nasdaq_filename = 'data/nasdaq_daily_data.csv'

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=2, suppress=True)


# contain
chart_df = pd.read_csv(filename)
CHART = CD.Chart(chart_df)
nasdaq_df = pd.read_csv(nasdaq_filename)
CHART.set_nasdaq_data(nasdaq_df)
CHART.set_data()

#月数、銘柄数を表示
print("number of day  :", end=" ")
print(CHART.date_length)
print("number of stocks :", end=" ")
print(CHART.row)

sp500_filename = 'data/sp500_daily_data.csv' #日次
sp500_df = pd.read_csv(sp500_filename)
MarketData = MK.Market(sp500_df)
MarketData.set()


#データ数チェッカー
if MarketData.length != CHART.chart_day_mat.shape[0]:
	print(MarketData.length)
	print(CHART.chart_day_mat.shape[0])
	print("DATA Length No Match")
	sys.exit(1)

####全銘柄に投資した場合
#mean_inv_sim_mat = mfiv.meanInv(CHART.chart_month_mat, CHART.mom_mat, start_month)
start_day=260
mean_inv_sim_mat = mfiv.meanInv(CHART.chart_day_mat, start_day)


#0:cash
#1:SPY
#2:inverse
#3:double inverse
result = np.delete(CHART.next_month_ex_return, [0,1,2,3], 1) #SPY, cash などを除く
num_of_stock = result.shape[1]
result = result.reshape(-1,1)
data_size = result.shape[0]
print(num_of_stock)
print(data_size)
#result = result.reshape(-1, num_of_stock)
#print(result.shape)

for i in range(len(CHART.stock_list)):
    list_item = CHART.stock_list[i]
    print('{0}:{1}'.format(i, list_item))


#モデルを読み込み
with open('model.pickle', mode='rb') as fp:
    model = pickle.load(fp)

filename = 'train_data/train_data.csv' #日次
df = pd.read_csv(filename)
X = df.values
X = np.delete(X, 0, axis=1) #インデックス列を削除
columns_list = df.columns.values.tolist()
del columns_list[:2]
print(X.shape)
Y = X[:,0] #学習用データとして抽出
X = X[:,1:] #ターゲットデータとして抽出

# 特徴量のスケーリング(非二値データ以外)
X_std = np.zeros((X.shape[0], 0))
sc = StandardScaler()
for j in range(X.shape[1]):
    if X[:,j].max()!=1 and X[:,j].min()!=0:#二値データでなかったら標準化を行う
        #sc.fit(X[:,j].reshape(-1,1))
        #X_j_std = sc.transform(X[:,j].reshape(-1,1))
        X_j_std = mfiv.standard_ex_outlier(X[:,j].reshape(-1,1), out=-999)
    else:
        X_j_std = X[:,j].reshape(-1,1)
    X_std = np.concatenate([X_std, X_j_std], axis=1)
#print(X_std[-1,:])

y = np.zeros((data_size, 1))
for i in range(X_std.shape[0]):
	if np.any(X_std[i,:] == -999):
		y[i,0]=-999
	else:
		a = model.predict(X_std[i,:].reshape(1,-1))
		val = a[0]
		y[i,0] = val
y = y.reshape(-1,num_of_stock)
print(y)
print(y.shape)
np.savetxt('train_data/predict_result.csv', y, delimiter=',')
