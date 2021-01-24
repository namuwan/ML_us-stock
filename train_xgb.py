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

import xgboost as xgb

import pickle
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=2, suppress=True)

market_file = 'data/sp500_daily_data.csv'
tnx_file ='data/us_yield_10years.csv'
df_market = pd.read_csv(market_file)
df_tnx = pd.read_csv(tnx_file)
market_chart = df_market.values
tnx_chart = df_tnx.values
market_chart = np.delete(market_chart, 0, axis=1) #インデックス列を削除
tnx_chart = np.delete(tnx_chart, 0, axis=1) #インデックス列を削除
market_chart = np.delete(market_chart, 0, axis=0) #インデックス列を削除
tnx_chart = np.delete(tnx_chart, 0, axis=0) #インデックス列を削除
market_chart = market_chart.flatten().astype(np.float32)
tnx_chart = tnx_chart.flatten().astype(np.float32)
log_market_chart = np.zeros(market_chart.size)
for i in range(market_chart.size):
    log_market_chart[i] = math.log10(market_chart[i]/market_chart[0])


#filename = 'train_data/train_data.csv'
#feature_slection.pyで選定した特徴量
filename = 'train_data/train_data_selected.csv'
#gdbtのimportanceの低いものは削除→あまりうまくいかない？
#filename = 'train_data/train_data_selected_for_gbdt.csv'


df = pd.read_csv(filename)
X = df.values
X = np.delete(X, 0, axis=1) #インデックス列を削除
columns_list = df.columns.values.tolist()
del columns_list[:2]

#カウント
feature_num = X.shape[1]-1
print("特徴量数", end=' ')
print(feature_num) #1は正解データ
timestamp_num = 4460
print("timestamp数", end=' ')
print(timestamp_num)
stock_num = (int)(X.shape[0]/timestamp_num)
print("銘柄数", end=' ')
print(stock_num)
print("総行数", end=' ')
print(timestamp_num * stock_num)
if X.shape[0] != timestamp_num*117:
    print("DATA Size no match")
    sys.exit(1)

#分割
#X_CV1, X_CV2, X_CV3 = np.split(X, 3)
#print(X_CV1.shape)
#print(X_CV2.shape)
#print(X_CV3.shape)

#欠損値(-999)を削除
#X = mfiv.delete_deficit_col(X, -999)
#print(X.shape)

#分割
split_ts_pos1 = 1983 #232000 #全タイムスタンプ数4460を分割して欠損値を除去したあと、ちょうど3分割くらいになる位置
split_ts_pos2 = 3376 #395000 #全タイムスタンプ数4460を分割して欠損値を除去したあと、ちょうど3分割くらいになる位置
split_ts_pos3 = 2697 #315500 #全タイムスタンプ数4460を分割して欠損値を除去したあと、ちょうど半分半分くらいになる位置
X_CV1, X_CV2, X_CV3 = np.array_split(X, [split_ts_pos1*stock_num, split_ts_pos2*stock_num])
X_CV4, X_CV5 = np.array_split(X, [split_ts_pos3*stock_num])
#X_CV1, X_CV2, X_CV3 = np.array_split(X, 3)
#X_CV4, X_CV5 = np.array_split(X, 2)

#"""
X_CV1 = mfiv.delete_deficit_col(X_CV1, -999)
X_CV2 = mfiv.delete_deficit_col(X_CV2, -999)
X_CV3 = mfiv.delete_deficit_col(X_CV3, -999)
X_CV4 = mfiv.delete_deficit_col(X_CV4, -999)
X_CV5 = mfiv.delete_deficit_col(X_CV5, -999)
X_all = mfiv.delete_deficit_col(X, -999)
#"""
print(X_CV1.shape)
print(X_CV2.shape)
print(X_CV3.shape)
print(X_CV4.shape)
print(X_CV5.shape)
print(X.shape)

#データ抽出
#Y = X[:,0] #ターゲットデータとして抽出
#X = X[:,1:] #学習データとして抽出
Y_CV1 = X_CV1[:,0] #ターゲットデータとして抽出
X_CV1 = X_CV1[:,1:] #学習データとして抽出
Y_CV2 = X_CV2[:,0] #ターゲットデータとして抽出
X_CV2 = X_CV2[:,1:] #学習データとして抽出
Y_CV3 = X_CV3[:,0] #ターゲットデータとして抽出
X_CV3 = X_CV3[:,1:] #学習データとして抽出"""
Y_CV4 = X_CV4[:,0] #ターゲットデータとして抽出
X_CV4 = X_CV4[:,1:] #学習データとして抽出"""
Y_CV5 = X_CV5[:,0] #ターゲットデータとして抽出
X_CV5 = X_CV5[:,1:] #学習データとして抽出"""
Y_all = X_all[:,0]
X_all = X_all[:,1:]

# 特徴量のスケーリング(非二値データ以外)
#X_std = mfiv.feature_standarization(X)
"""
X_CV1 = mfiv.feature_standarization(X_CV1)
X_CV2 = mfiv.feature_standarization(X_CV2)
X_CV3 = mfiv.feature_standarization(X_CV3)
"""

#訓練データと学習データを分割
#X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size=0.25, shuffle=False)

#print("訓練データ数", end=' ')
#print(X_train.shape)
#print(X_CV1.shape)
#hoge

#重回帰
#model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
#L2正則化
#model = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, random_state=None)
#勾配Boosting
#model_cv1 = ExtraTreesRegressor(
#model_cv1 = GradientBoostingRegressor(
"""
clf = xgb.XGBRegressor()
clf_cv = GridSearchCV(clf, {'max_depth':[2,4,6], 'n_estimators':[50, 100, 200]}, verbose=1)
clf_cv.fit(X_all, Y_all)
print(clf_cv.best_params_, clf_cv.best_score_)
hoge
"""

params = {
    'eta':0.05,
    'max_depth':3,
    'min_child_weight':5,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'alpha':0.2,
    'nthread':4,
    'silent':True
}

model_cv1 = xgb.XGBRegressor(**params)
model_cv2 = xgb.XGBRegressor(**params)
model_cv4 = xgb.XGBRegressor(**params)
model_all = xgb.XGBRegressor(**params)

"""
#paramJ1_1 = {'n_estimators': list(range(20, 101, 10)),
#             'learning_rate': list(np.arange(0.05, 0.20, 0.01))}
paramJ1_2 = {'min_samples_split': list(range(1000, 10001, 1000)),
             'min_samples_leaf': list(np.arange(1000, 10001, 1000))}
gsearch1 = GridSearchCV(estimator = model,
    param_grid = paramJ1_2,
    cv = 2,#cross_validation数
    n_jobs=-1,#-1：コア数で実行
    scoring = 'neg_mean_squared_error')
gsearch1.fit(X_train, Y_train)
cv_result = pd.DataFrame(gsearch1.cv_results_)
#cv_result = cv_result[['param_n_estimators', 'param_learning_rate', 'mean_test_score']]
cv_result = cv_result[['param_min_samples_split', 'param_min_samples_leaf', 'mean_test_score']]
#cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_n_estimators', 'param_learning_rate')
cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_min_samples_split', 'param_min_samples_leaf')
sns.heatmap(cv_result_pivot, cmap='Greys', annot=True)
plt.show()
# test精度の平均が最も高かった組み合わせを出力
hoge
print(gsearch1.best_params_)
#"""

#model.fit(X_train, Y_train)
#df_coefficient = pd.DataFrame(model.coef_, columns=["係数"], index=columns_list)
#print(df_coefficient)
#print(model.coef_)
#print(model.intercept_)
#print(model.score(X_test, Y_test))
#print(model.score(X_CV2, Y_CV2))

print()

model_cv1.fit(X_CV1, Y_CV1)
train_coef_cv1 = np.corrcoef(model_cv1.predict(X_CV1), Y_CV1.reshape(1,-1)) # 予測モデルの相関行列を計算
print("訓練_相関係数_CV1")
print('{:.4f}'.format(train_coef_cv1[0,1])) # 相関行列を表示

model_cv2.fit(X_CV2, Y_CV2)
train_coef_cv2 = np.corrcoef(model_cv2.predict(X_CV2), Y_CV2.reshape(1,-1)) # 予測モデルの相関行列を計算
print("訓練_相関係数_CV2")
print('{:.4f}'.format(train_coef_cv2[0,1])) # 相関行列を表示

model_cv4.fit(X_CV4, Y_CV4)
train_coef_cv4 = np.corrcoef(model_cv4.predict(X_CV4), Y_CV4.reshape(1,-1)) # 予測モデルの相関行列を計算
print("訓練_相関係数_CV4")
print('{:.4f}'.format(train_coef_cv4[0,1])) # 相関行列を表示

model_all.fit(X_all, Y_all)
train_coef_all = np.corrcoef(model_all.predict(X_all), Y_all.reshape(1,-1)) # 予測モデルの相関行列を計算
print("訓練_相関係数_ALL")
print('{:.4f}'.format(train_coef_all[0,1])) # 相関行列を表示

print()

test_coef_1_2 = np.corrcoef(model_cv1.predict(X_CV2), Y_CV2.reshape(1,-1)) # 予測モデルの相関行列を計算
print("テスト_相関係数_CV1 -> CV2")
print('{:.4f}'.format(test_coef_1_2[0,1])) # 相関行列を表示

test_coef_1_3 = np.corrcoef(model_cv1.predict(X_CV3), Y_CV3.reshape(1,-1)) # 予測モデルの相関行列を計算
print("テスト_相関係数_CV1 -> CV3")
print('{:.4f}'.format(test_coef_1_3[0,1])) # 相関行列を表示

test_coef_2_3 = np.corrcoef(model_cv2.predict(X_CV3), Y_CV3.reshape(1,-1)) # 予測モデルの相関行列を計算
print("テスト_相関係数_CV2 -> CV3")
print('{:.4f}'.format(test_coef_2_3[0,1])) # 相関行列を表示

test_coef_4_5 = np.corrcoef(model_cv4.predict(X_CV5), Y_CV5.reshape(1,-1)) # 予測モデルの相関行列を計算
print("テスト_相関係数_CV4 -> CV5")
print('{:.4f}'.format(test_coef_4_5[0,1])) # 相関行列を表示



cumsum_r_1_2 = np.zeros(timestamp_num)
cumsum_r_1_3 = np.zeros(timestamp_num)
cumsum_r_2_3 = np.zeros(timestamp_num)
cumsum_r_4_5 = np.zeros(timestamp_num)
cumsum_r_all = np.zeros(timestamp_num)
for i in range(split_ts_pos1,split_ts_pos2):
    X_def_deleted = mfiv.delete_deficit_col(X[i*stock_num:(i+1)*stock_num,:], -999)
    if X_def_deleted.shape[0]!=0: #deficitで行数0になる可能性がある
        y_predicted = model_cv1.predict(X_def_deleted[:,1:])
        y_result = X_def_deleted[:,0]
        coef = np.corrcoef(y_predicted, y_result)[0,1]
        cumsum_r_1_2[i] = cumsum_r_1_2[i-1] + coef
cumsum_r_1_2 = np.trim_zeros(cumsum_r_1_2, "b") #末尾ゼロを削除
for i in range(split_ts_pos2,timestamp_num):
    X_def_deleted = mfiv.delete_deficit_col(X[i*stock_num:(i+1)*stock_num,:], -999)
    if X_def_deleted.shape[0]!=0: #deficitで行数0になる可能性がある
        y_predicted = model_cv1.predict(X_def_deleted[:,1:])
        y_result = X_def_deleted[:,0]
        coef = np.corrcoef(y_predicted, y_result)[0,1]
        cumsum_r_1_3[i] = cumsum_r_1_3[i-1] + coef
cumsum_r_1_3 = np.trim_zeros(cumsum_r_1_3, "b")
for i in range(split_ts_pos2,timestamp_num):
    X_def_deleted = mfiv.delete_deficit_col(X[i*stock_num:(i+1)*stock_num,:], -999)
    if X_def_deleted.shape[0]!=0: #deficitで行数0になる可能性がある
        y_predicted = model_cv2.predict(X_def_deleted[:,1:])
        y_result = X_def_deleted[:,0]
        coef = np.corrcoef(y_predicted, y_result)[0,1]
        cumsum_r_2_3[i] = cumsum_r_2_3[i-1] + coef
cumsum_r_2_3 = np.trim_zeros(cumsum_r_2_3, "b")
for i in range(split_ts_pos3, timestamp_num):
    X_def_deleted = mfiv.delete_deficit_col(X[i*stock_num:(i+1)*stock_num,:], -999)
    if X_def_deleted.shape[0]!=0: #deficitで行数0になる可能性がある
        y_predicted = model_cv4.predict(X_def_deleted[:,1:])
        y_result = X_def_deleted[:,0]
        coef = np.corrcoef(y_predicted, y_result)[0,1]
        cumsum_r_4_5[i] = cumsum_r_4_5[i-1] + coef
cumsum_r_4_5 = np.trim_zeros(cumsum_r_4_5, "b")
for i in range(0, timestamp_num):
    X_def_deleted = mfiv.delete_deficit_col(X[i*stock_num:(i+1)*stock_num,:], -999)
    if X_def_deleted.shape[0]!=0: #deficitで行数0になる可能性がある
        y_predicted = model_all.predict(X_def_deleted[:,1:])
        y_result = X_def_deleted[:,0]
        coef = np.corrcoef(y_predicted, y_result)[0,1]
        cumsum_r_all[i] = cumsum_r_all[i-1] + coef
cumsum_r_all = np.trim_zeros(cumsum_r_all, "b")
print(cumsum_r_all.shape)

plt.subplot(2,1,1)
plt.plot(cumsum_r_1_2, label='cv1to2')
plt.plot(cumsum_r_1_3, label='cv1to3')
plt.plot(cumsum_r_2_3, label='cv2to3')
plt.plot(cumsum_r_4_5, label='cv4to5')
#plt.plot(cumsum_r_all, label='all')
plt.legend()

ax1 = plt.subplot(2,1,2)
ax1.plot(log_market_chart, color='blue', label='log_sp500')
ax2 = ax1.twinx()
ax2.plot(tnx_chart, color='red', label='us_10years_yield')
plt.show()
hoge



#予測モデルの相関係数を計算
#coef = np.corrcoef(X_std[:,1], Y.reshape(1,-1)) # 相関行列を計算
#train_coef = np.corrcoef(model.predict(X_train), Y_train.reshape(1,-1)) # 予測モデルの相関行列を計算
#test_coef = np.corrcoef(model.predict(X_test), Y_test.reshape(1,-1)) # 予測モデルの相関行列を計算

df_importance_cv1 = pd.DataFrame(model_cv1.feature_importances_, index=columns_list)
df_importance_cv2 = pd.DataFrame(model_cv2.feature_importances_, index=columns_list)
df_importance_cv4 = pd.DataFrame(model_cv4.feature_importances_, index=columns_list)
df_importance_all = pd.DataFrame(model_all.feature_importances_, index=columns_list)
print(df_importance_cv1)
print(df_importance_cv2)
print(df_importance_cv4)
print(df_importance_all)
hoge
"""
n_features = X.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center') # 横棒グラフ
plt.yticks(np.arange(n_features), columns_list)
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
"""

#print(model.predict(X_test))
print(model.predict(X_CV1))
plt.scatter(model.predict(X_test), Y_test, color='blue', s=0.1) # 説明変数と目的変数のデータ点の散布図をプロット
#plt.scatter(model.predict(X_std[:,1].reshape(-1,1)), Y, color = 'blue', s=0.1)  # 説明変数と目的変数のデータ点の散布図をプロット
#plt.show()


#モデルを保存
with open('model.pickle', mode='wb') as fp:
    pickle.dump(model, fp)
