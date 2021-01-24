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

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=2, suppress=True)

file_predict_result = 'train_data/predict_result.csv'

y_predicted = np.loadtxt('train_data/predict_result.csv', delimiter=',')
#print(predict_result)
#print(predict_result.shape)
date_num = y_predicted.shape[0]
stock_num = y_predicted.shape[1]

filename = 'train_data/train_data.csv'

df = pd.read_csv(filename)
X = df.values
X = np.delete(X, 0, axis=1) #インデックス列を削除
y_result = X[:,0] #学習用データとして抽出
y_result = y_result.reshape(date_num,-1)

print(y_predicted.shape)
print(y_result.shape)

#y_sign：アップorダウンが正解だったら1 不正解だったら-1
#各timestampで全銘柄のy_signのsumを計算
#それを累積させて推移を見る
y_sign_cumsum = np.zeros((date_num,1))
for i in range(date_num):
    y_sign_sum = 0
    for j in range(stock_num):
        if y_predicted[i][j]!=-999 and y_result[i][j]!=-999:
            if y_predicted[i][j]*y_result[i][j]>0:
                y_sign_sum = y_sign_sum + 1
            else:
                y_sign_sum = y_sign_sum - 1
    y_sign_cumsum[i][0] = y_sign_cumsum[i-1][0] + y_sign_sum
print(y_sign_cumsum)
plt.plot(y_sign_cumsum)
plt.show()



#columns_list = df.columns.values.tolist()
#del columns_list[:2]
