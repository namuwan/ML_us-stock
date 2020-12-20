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


filename = 'train_data/train_data.csv'
#貢献がない特徴量を削除
#filename = 'train_data/train_data_selected.csv'

df = pd.read_csv(filename)
X = df.values
X = np.delete(X, 0, axis=1) #インデックス列を削除
columns_list = df.columns.values.tolist()
del columns_list[:2]

#欠損値(-999)を削除
missing_list = []
there_is_deficit = 0
col = X.shape[0]
row = X.shape[1]
for i in range(col):
    for j in range(row):
        if X[i][j]==-999:
            there_is_deficit = 1
    if there_is_deficit==1:
        missing_list.append(i)
    there_is_deficit = 0
X = np.delete(X, missing_list, axis=0)


#データ抽出
Y = X[:,0] #学習用データとして抽出
X = X[:,1:] #ターゲットデータとして抽出


# 特徴量のスケーリング(非二値データ以外)
X_std = np.zeros((X.shape[0], 0))
sc = StandardScaler()
for j in range(X.shape[1]):
    if X[:,j].max()!=1 and X[:,j].min()!=0:#二値データでなかったら標準化を行う
        #print(X[:,j])
        #print(X[:,j].shape)
        #print(X[:,j].ndim)
        sc.fit(X[:,j].reshape(-1,1))
        X_j_std = sc.transform(X[:,j].reshape(-1,1))
    else:
        X_j_std = X[:,j].reshape(-1,1)
    X_std = np.concatenate([X_std, X_j_std], axis=1)


#訓練データと学習データを分割
X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size=0.25, shuffle=False)
print("訓練データ数", end=' ')
print(X_train.size)


#重回帰
#model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
#L2正則化
#model = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, random_state=None)
#L1正則化
#model = Lasso(alpha=0.05, fit_intercept=True, normalize=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection="cyclic")
#勾配Boosting
model = GradientBoostingRegressor(
    min_samples_split = 2000,
    min_samples_leaf = 1000,
    max_depth = 3,
    max_features = 'sqrt',
    subsample = 0.8,
    n_estimators=50,
    learning_rate=0.05)
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

model.fit(X_train, Y_train)
#df_coefficient = pd.DataFrame(model.coef_, columns=["係数"], index=columns_list)
#print(df_coefficient)
#model.fit(X_std[:,1].reshape(-1,1), Y)
#print(model.coef_)
#print(model.intercept_)
print(model.score(X_test, Y_test))
#print(model.score(X_std[:,1].reshape(-1,1), Y))

#予測モデルの相関係数を計算
#coef = np.corrcoef(X_std[:,1], Y.reshape(1,-1)) # 相関行列を計算
train_coef = np.corrcoef(model.predict(X_train), Y_train.reshape(1,-1)) # 予測モデルの相関行列を計算
print("訓練_相関係数")
print(train_coef) # 相関行列を表示
test_coef = np.corrcoef(model.predict(X_test), Y_test.reshape(1,-1)) # 予測モデルの相関行列を計算
print("テスト_相関係数")
print(test_coef) # 相関行列を表示

df_feature_importance = pd.DataFrame(model.feature_importances_, index=columns_list)
print(df_feature_importance)
"""
n_features = X.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center') # 横棒グラフ
plt.yticks(np.arange(n_features), columns_list)
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
"""

print(model.predict(X_test))
plt.scatter(model.predict(X_test), Y_test, color='blue', s=0.1) # 説明変数と目的変数のデータ点の散布図をプロット
#plt.scatter(model.predict(X_std[:,1].reshape(-1,1)), Y, color = 'blue', s=0.1)  # 説明変数と目的変数のデータ点の散布図をプロット
#plt.show()


#モデルを保存
with open('model.pickle', mode='wb') as fp:
    pickle.dump(model, fp)
