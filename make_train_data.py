#coding:utf-8
########################
#choice asset
########################

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

filename = 'data/stock_daily_data.csv' #日次
nasdaq_filename = 'data/nasdaq_daily_data.csv'
tnx_filename = 'data/us_yield_10years.csv'

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=2, suppress=True)

start_month = 12

# contain
chart_df = pd.read_csv(filename)
CHART = CD.Chart(chart_df)
nasdaq_df = pd.read_csv(nasdaq_filename)
CHART.set_nasdaq_data(nasdaq_df)
CHART.set_data()

#金利データをセット
tnx_df = pd.read_csv(tnx_filename)
CHART.set_tnx(tnx_df)


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


#複数月荷重mixランキング
"""
mPara1 = 3; w1 = 0.6
mPara2 = 6; w2 = 0.3
mPara3 = 11; w3 = 0.3
CHART.set_rank_point_mat(mPara1, mPara2, mPara3, w1, w2, w3)
"""

"""
dPara1 = 3*20; w1=0.6
dPara2 = 6*20; w2=0.3
dPara3 = 11*20; w3=0.3
CHART.set_daily_rank_mat(dPara1, dPara2, dPara3, w1, w2, w3)
#print(CHART.daily_rank_mat)

#Buy or Sell 判断
mov_ave_para = 11;
buy_thresh = 1.0
sell_thresh = 0.9
CHART.set_mov_ave_mat(mov_ave_para)
CHART.set_bs_mat(buy_thresh, sell_thresh)

mov_ave_para = 20*11
buy_thresh = 1.0
sell_thresh = 1.0
CHART.set_daily_mov_ave_mat(mov_ave_para)
CHART.set_daily_bs_mat(buy_thresh, sell_thresh)
#print(CHART.daily_bs_buy_mat)
#print(CHART.daily_bs_sell_mat)
"""

"""
clf = linear_model.LinearRegression()
X = CHART.chart_day_mat[-200:-1,1].reshape(-1,1) #SPY
X = X/X[0]*100
Y = CHART.chart_day_mat[-200:-1,4].reshape(-1,1) #GOOG
Y = Y/Y[0]*100
clf.fit(X, Y)
print(clf.coef_)#回帰係数
print(clf.intercept_)#切片 (誤差)
print(r2_score(X, Y))#決定係数
#print(clf.score(X, Y))
plt.scatter(X, Y, s=1)
plt.plot(X, clf.predict(X))
#plt.plot([0,10],[-100,100],c="red")
#plt.show()
"""

missing_list = []

result = np.delete(CHART.next_month_ex_return, [0,1,2,3], 1) #SPY, cash などを除く
result = result.reshape(-1,1)

"""
#ランキング
rank_value = np.delete(CHART.daily_rank_mat, [0,1,2,3], 1) #SPY, cashなどを除く
rank_value = rank_value.reshape(-1,1)
"""

"""
#n日前からの超過リターン
feat01 = CHART.ret_n_day_ex_return(220)
feat02 = CHART.ret_n_day_ex_return(180)
feat03 = CHART.ret_n_day_ex_return(120)
feat04 = CHART.ret_n_day_ex_return(60)
feat05 = CHART.ret_n_day_ex_return(20)
feat06 = CHART.ret_n_day_ex_return(10)
data = np.concatenate([result, feat01, feat02, feat03, feat04, feat05, feat06], 1)
df = pd.DataFrame(data=data)
df.columns = ['result', 'feat01','feat02','feat03','feat04','feat05','feat06']
df.to_csv("train_data/temp_train_data.csv")
hoge
"""

"""
#過去n日の最大値からの下落率
feat01 = CHART.ret_n_day_drop(220)
feat02 = CHART.ret_n_day_drop(180)
feat03 = CHART.ret_n_day_drop(120)
feat04 = CHART.ret_n_day_drop(60)
feat05 = CHART.ret_n_day_drop(20)
feat06 = CHART.ret_n_day_drop(10)
data = np.concatenate([result, feat01, feat02, feat03, feat04, feat05, feat06], 1)
df = pd.DataFrame(data=data)
df.columns = ['result', 'feat01','feat02','feat03','feat04','feat05','feat06']
df.to_csv("train_data/temp_train_data.csv")
hoge
"""

"""
#過去n日高値から何日経過しているか
feat01 = CHART.ret_n_day_under(220)
feat02 = CHART.ret_n_day_under(180)
feat03 = CHART.ret_n_day_under(120)
feat04 = CHART.ret_n_day_under(60)
feat05 = CHART.ret_n_day_under(20)
feat06 = CHART.ret_n_day_under(10)
data = np.concatenate([result, feat01, feat02, feat03, feat04, feat05, feat06], 1)
df = pd.DataFrame(data=data)
df.columns = ['result', 'feat01','feat02','feat03','feat04','feat05','feat06']
df.to_csv("train_data/temp_train_data.csv")
hoge
"""

#"""
#新高値圏からn日以内にいるか
feat01 = CHART.ret_new_highs(60)
feat02 = CHART.ret_new_highs(20)
data = np.concatenate([result, feat01, feat02], 1)
df = pd.DataFrame(data=data)
df.columns = ['result', 'feat01', 'feat02']
df.to_csv("train_data/temp_train_data.csv")
hoge
#"""

"""
#n日安値からの上昇率
feat01 = CHART.ret_n_day_bottom_rise(220)
feat02 = CHART.ret_n_day_bottom_rise(180)
feat03 = CHART.ret_n_day_bottom_rise(120)
feat04 = CHART.ret_n_day_bottom_rise(60)
feat05 = CHART.ret_n_day_bottom_rise(20)
feat06 = CHART.ret_n_day_bottom_rise(10)
data = np.concatenate([result, feat01, feat02, feat03, feat04, feat05, feat06], 1)
df = pd.DataFrame(data=data)
df.columns = ['result', 'feat01','feat02','feat03','feat04','feat05','feat06']
df.to_csv("train_data/temp_train_data.csv")
hoge
"""

"""
#n日移動平均線からの乖離率
feat01 = CHART.ret_div_sma(220)
feat02 = CHART.ret_div_sma(180)
feat03 = CHART.ret_div_sma(120)
feat04 = CHART.ret_div_sma(60)
feat05 = CHART.ret_div_sma(20)
feat06 = CHART.ret_div_sma(10)
data = np.concatenate([feat01, feat02, feat03, feat04, feat05, feat06], 1)
df = pd.DataFrame(data=data)
df.columns = ['feat01','feat02','feat03','feat04','feat05','feat06']
df.to_csv("train_data/temp_train_data.csv")
hoge
"""

"""
#n日移動平均線の上か下か
feat01 = CHART.ret_above_sma(220)
feat02 = CHART.ret_above_sma(180)
feat03 = CHART.ret_above_sma(120)
feat04 = CHART.ret_above_sma(60)
feat05 = CHART.ret_above_sma(20)
feat06 = CHART.ret_above_sma(10)
data = np.concatenate([feat01, feat02, feat03, feat04, feat05, feat06], 1)
df = pd.DataFrame(data=data)
df.columns = ['feat01','feat02','feat03','feat04','feat05','feat06']
df.to_csv("train_data/temp_train_data.csv")
hoge
"""

"""
#直近の動き
feat01 = CHART.ret_n_day_ex_return(3)
feat02 = CHART.ret_n_day_ex_return(2)
feat03 = CHART.ret_n_day_ex_return(1)
data = np.concatenate([feat01, feat02, feat03], 1)
df = pd.DataFrame(data=data)
df.columns = ['feat01', 'feat02', 'feat03']
df.to_csv("train_data/temp_train_data.csv")
hoge
"""

"""
#ボラティリティ
feat01 = CHART.ret_volatility(220)
feat02 = CHART.ret_volatility(180)
feat03 = CHART.ret_volatility(120)
feat04 = CHART.ret_volatility(60)
feat05 = CHART.ret_volatility(20)
feat06 = CHART.ret_volatility(10)
data = np.concatenate([feat01, feat02, feat03, feat04, feat05, feat06], 1)
df = pd.DataFrame(data=data)
df.columns = ['feat01', 'feat02', 'feat03', 'feat04', 'feat05', 'feat06']
df.to_csv("train_data/temp_train_data.csv")
hoge
"""

#シャープレシオ
feat01 = CHART.ret_sharpe_ratio()
df = pd.DataFrame(data=feat01)
df.columns = ['feat01']
df.to_csv("train_data/temp_train_data.csv")
hoge



"""
#金利
feat01 = CHART.ret_tnx_data()
df = pd.DataFrame(data=feat01)
df.columns = ['feat01']
df.to_csv("train_data/temp_train_data.csv")
hoge
"""
"""
#金利 n day return
mat = CHART.tnx_mat
feat01 = CHART.ret_market_return(mat, 220)
feat02 = CHART.ret_market_return(mat, 180)
feat03 = CHART.ret_market_return(mat, 120)
feat04 = CHART.ret_market_return(mat, 60)
feat05 = CHART.ret_market_return(mat, 20)
feat06 = CHART.ret_market_return(mat, 10)
data = np.concatenate([feat01, feat02, feat03, feat04, feat05, feat06], 1)
df = pd.DataFrame(data=data)
df.columns = ['feat01', 'feat02', 'feat03', 'feat04', 'feat05', 'feat06']
df.to_csv("train_data/temp_train_data.csv")
"""

"""
#ゴールデンクロス
feat01 = CHART.ret_gld_cross(220)
feat02 = CHART.ret_gld_cross(180)
feat03 = CHART.ret_gld_cross(120)
feat04 = CHART.ret_gld_cross(60)
feat05 = CHART.ret_gld_cross(20)
feat06 = CHART.ret_gld_cross(10)
"""

"""
for i in range(result.shape[0]):
	if result[i][0]==-999:
		missing_list.append(i)
	elif rank_value[i][0]==0:
		missing_list.append(i)
	elif day_220_ex_return[i][0]==-999:
		missing_list.append(i)
	elif day_180_ex_return[i][0]==-999:
		missing_list.append(i)
	elif day_120_ex_return[i][0]==-999:
		missing_list.append(i)
	elif day_60_ex_return[i][0]==-999:
		missing_list.append(i)
	elif day_20_ex_return[i][0]==-999:
		missing_list.append(i)
	elif day_10_ex_return[i][0]==-999:
		missing_list.append(i)
	elif day_220_drop[i][0]==-999:
		missing_list.append(i)
	elif day_180_drop[i][0]==-999:
		missing_list.append(i)
	elif day_120_drop[i][0]==-999:
		missing_list.append(i)
	elif day_60_drop[i][0]==-999:
		missing_list.append(i)
	elif day_20_drop[i][0]==-999:
		missing_list.append(i)
	elif day_10_drop[i][0]==-999:
		missing_list.append(i)
	elif day_220_under[i][0]==-999:
		missing_list.append(i)
	elif day_180_under[i][0]==-999:
		missing_list.append(i)
	elif day_120_under[i][0]==-999:
		missing_list.append(i)
	elif day_60_under[i][0]==-999:
		missing_list.append(i)
	elif day_20_under[i][0]==-999:
		missing_list.append(i)
	elif day_10_under[i][0]==-999:
		missing_list.append(i)
	elif new_highs[i][0]==-999:
		missing_list.append(i)
	elif feat01[i][0]==-999:
		missing_list.append(i)
	elif feat02[i][0]==-999:
		missing_list.append(i)
	elif feat03[i][0]==-999:
		missing_list.append(i)
	elif feat04[i][0]==-999:
		missing_list.append(i)
	elif feat05[i][0]==-999:
		missing_list.append(i)
	elif feat06[i][0]==-999:
		missing_list.append(i)
"""

"""
result = np.delete(result, missing_list, 0)
rank_value = np.delete(rank_value, missing_list, 0)
day_220_ex_return = np.delete(day_220_ex_return, missing_list, 0)
day_180_ex_return = np.delete(day_180_ex_return, missing_list, 0)
day_120_ex_return = np.delete(day_120_ex_return, missing_list, 0)
day_60_ex_return = np.delete(day_60_ex_return, missing_list, 0)
day_20_ex_return = np.delete(day_20_ex_return, missing_list, 0)
day_10_ex_return = np.delete(day_10_ex_return, missing_list, 0)
day_220_drop = np.delete(day_220_drop, missing_list, 0)
day_180_drop = np.delete(day_180_drop, missing_list, 0)
day_120_drop = np.delete(day_120_drop, missing_list, 0)
day_60_drop = np.delete(day_60_drop, missing_list, 0)
day_20_drop = np.delete(day_20_drop, missing_list, 0)
day_10_drop = np.delete(day_10_drop, missing_list, 0)
day_220_under = np.delete(day_220_under, missing_list, 0)
day_180_under = np.delete(day_180_under, missing_list, 0)
day_120_under = np.delete(day_120_under, missing_list, 0)
day_60_under = np.delete(day_60_under, missing_list, 0)
day_20_under = np.delete(day_20_under, missing_list, 0)
day_10_under = np.delete(day_10_under, missing_list, 0)
new_highs = np.delete(new_highs, missing_list, 0)
feat01 = np.delete(feat01, missing_list, 0)
feat02 = np.delete(feat02, missing_list, 0)
feat03 = np.delete(feat03, missing_list, 0)
feat04 = np.delete(feat04, missing_list, 0)
feat05 = np.delete(feat05, missing_list, 0)
feat06 = np.delete(feat06, missing_list, 0)
"""

data = np.concatenate([result, feat01, feat02, feat03, feat04, feat05, feat06], 1)
df = pd.DataFrame(data=data)
df.columns = ['result', 'feat01','feat02','feat03','feat04','feat05','feat06']
df['log_result'] = (np.log10(df['result']/100+1+1)-0.3)*100
print(df)
print(df.corr().loc[:,['result','log_result']])


"""
#散布図
df.plot.scatter(x='feat07', y='result')
plt.show()
"""
"""
#ヒスト
df['log_result'].hist(bins=300)
print(df.describe())
plt.show()
"""
#"""
#bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 4.0, 6.0, 30]
#bins = [-0.01, 0.5, 1.0]
bins = [0.0, 5, 10, 20, 30, 50, 80, 100, 150, 200, 500]
df['band'] = pd.cut(df['feat05'], bins=bins)
print(df['band'].value_counts(sort=False))
disp_df = df[['band', 'result']].groupby(['band'], as_index=False).describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
print(disp_df)
#"""

"""
bins_n_return = [-100,-50,-30,-20,-10,-5,-3,0,3,5,10,20,30,50,100,500]
df['n_return_band'] = pd.cut(df['n_day_ex_return'], bins=bins_n_return)
print(df['n_return_band'].value_counts(sort=False))
disp_df = df[['n_return_band', 'result']].groupby(['n_return_band'], as_index=False).describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
"""



#"""
# sklearn.linear_model.LinearRegression クラスを読み込み
"""
from sklearn import linear_model
from sklearn.metrics import r2_score
clf = linear_model.LinearRegression()
X = rank_value
clf.fit(X, Y)
# 回帰係数
print(clf.coef_)
# 切片 (誤差)
print(clf.intercept_)
# 決定係数
print(r2_score(X, Y))
#print(clf.score(X, Y))
#"""
"""
#scatter
plt.scatter(X, Y, s=1)
plt.plot(X, clf.predict(X))
plt.plot([0,10],[-100,100],c="red")
plt.show()
#histgram
#plt.hist(X, bins=25, alpha=0.3, histtype='stepfilled', color='r')
#plt.hist(Y, bins=25, alpha=0.3, histtype='stepfilled', color='b')
#plt.show()
"""
"""
result_mom = np.empty(0)
mean_mom = np.empty(0)
for i in range(1,len(result)):
	result_mom = np.append(result_mom, result[i]/result[i-1]*100-100)
result_mom = np.delete(result_mom, np.s_[:12])
for i in range(1,len(mean_inv_sim_mat)):
	mean_mom = np.append(mean_mom, mean_inv_sim_mat[i]/mean_inv_sim_mat[i-1]*100-100)
print(len(result_mom))
print(result_mom)
#print(len(mean_mom))
print(mean_mom)

chart_mat_SP500 = np.delete(chart_mat_SP500, np.s_[:12])
sp500_mom = np.empty(0)
for i in range(1, len(chart_mat_SP500)):
	sp500_mom = np.append(sp500_mom, chart_mat_SP500[i]/chart_mat_SP500[i-1]*100-100)
print(len(sp500_mom))
print(sp500_mom)
"""

"""
# sklearn.linear_model.LinearRegression クラスを読み込み
from sklearn import linear_model
clf = linear_model.LinearRegression()
X=mean_mom.reshape(-1,1)
#X=sp500_mom.reshape(-1,1)
#Y=mean_mom.reshape(-1,1)
Y=result_mom.reshape(-1,1)
clf.fit(X, Y)
# 回帰係数
print(clf.coef_)
# 切片 (誤差)
print(clf.intercept_)
# 決定係数
print(clf.score(X, Y))

#scatter
plt.scatter(mean_mom, result_mom)
plt.plot(X, clf.predict(X))
#plt.scatter(sp500_mom, mean_mom)
#plt.plot([-12,20],[-12,20],c="red")
#plt.scatter(sp500_mom, result_mom)
plt.plot([-20,20],[-20,20],c="red")
#histgram
#plt.hist(result_mom, bins=25, alpha=0.3, histtype='stepfilled', color='r')
#plt.hist(mean_mom, bins=25, alpha=0.3, histtype='stepfilled', color='b')

#plt.show()
"""

"""
chart_t = mom_mat[-12:].T - np.ones((mom_mat[-12:].T.shape[0], mom_mat[-12:].T.shape[1]))
chart_t = np.array([n*100 for n in chart_t])
for i in range(chart_t.shape[0]):
	for j in range(chart_t.shape[1]):
		if chart_t[i][j] < -99:
			chart_t[i][j] = 0

#for i in range(len(stock_list)):
#	print(str(i) + " " + stock_list[i])
#偏相関係数行列(SP500影響を除く)
partial_coef = np.zeros((len(stock_list), len(stock_list)))
for i in range(len(stock_list)):
	for j in range(len(stock_list)):
		r12 = np.corrcoef(chart_t[i],chart_t[j])[0,1]
		r13 = np.corrcoef(chart_t[i],chart_t[76])[0,1]#no.76=sp500
		r23 = np.corrcoef(chart_t[j],chart_t[76])[0,1]
		partial_coef[i][j] = (r12-r13*r23) / ( ((1-r13**2)**(1/2)) * ((1-r23**2)**(1/2)) )
for i in range(partial_coef.shape[0]):
	for j in range(partial_coef.shape[1]):
		if partial_coef[i][j]<0.5:
			partial_coef[i][j]=0
print(partial_coef)
"""
"""
fig, ax = plt.subplots()
sns.heatmap(partial_coef, vmin=-1, vmax=1, ax=ax, cmap='YlGnBu', xticklabels=stock_list, yticklabels=stock_list)
ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=6)
plt.show()
"""



"""
### Display Result %%%%%%%%%
for i=1:rowSize
	assetName = returnAsset(rankIndex(columnSize,i));
	%rankScore = movAvereturnMat(1,rankIndex(columnSize,i));
	rankScore = rankPointMat(columnSize,rankIndex(columnSize,i));
	bsScore = movAveBSMat2(1, rankIndex(columnSize,i)) / movAveBSMat(columnSize,rankIndex(columnSize,i));
	if bsMat(1,rankIndex(columnSize,i))==1
	 borsstr = 'Buy ';
	else
	 borsstr = 'Sell';
	end
	fprintf('%2d  %s  %s  rank score:%6.3f  bs score:%.3f\n', i, assetName, borsstr, rankScore, bsScore);
end
"""
