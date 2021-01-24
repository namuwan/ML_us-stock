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


np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.max_rows', None)

filename = 'train_data/train_data.csv'

df = pd.read_csv(filename)
X = df.values
X = np.delete(X, 0, axis=1) #インデックス列を削除
columns_list = df.columns.values.tolist()
del columns_list[:2]
#print(columns_list)

#欠損値を削除
X = mfiv.delete_deficit_col(X, -999)
print(X.shape)

feature_num = X.shape[1]-1
r2_mat = np.zeros((feature_num,1))
for i in range(feature_num):
	#r = np.corrcoef(X[:,0], abs(X[:,i+1]))[0,1]
	r = np.corrcoef(X[:,0], X[:,i+1])[0,1]
	r2_mat[i,0] = r*r*100
print(r2_mat)
#hoge

df_r2 = pd.DataFrame(r2_mat, columns=['r2'], index=columns_list)
print(df_r2)
hoge
#df = pd.DataFrame(r2_mat, columns=['r2'])
#print(df)
#df_s = df.sort_values('r2', ascending=False)
#print(df_s)
df_r2_selected = df_r2[df_r2['r2'] > 0.03*0.03*100]
print(df_r2_selected)
hoge

selected_feature_list = df_r2_selected.index.values.tolist()
for i in range(len(selected_feature_list)):
	selected_feature_list[i] = 'abs_' + selected_feature_list[i]
#selected_feature_list.insert(0, 'result')
#print(selected_feature_list)

df = df.rename(columns=lambda s: 'abs_'+s)
df_selected = df.loc[:,selected_feature_list].abs()
df_selected = df_selected.replace(999,-999)
print(df_selected.head())

df_selected.to_csv("train_data/train_data_selected_temp.csv")
