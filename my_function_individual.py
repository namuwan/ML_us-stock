import InvSim as IS

import pandas as pd
import numpy as np
from scipy import signal
#import keras.backend as K

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as dates

from statistics import stdev

FILL_DEFICIT = 99999


#pandasからnumpyに変換 月次チャート
def contain_month_chart(chart_df):

	stock_list = list(chart_df.columns.values)
	del stock_list[0]

	date_mat = chart_df.iloc[:,0:1].values #numpy配列に変換
	chart_mat = chart_df.values
	chart_mat = np.delete(chart_mat, 0, 1)
	chart_mat = np.delete(chart_mat, 0, 0)
	chart_mat = chart_mat.astype(np.float32)
	chart_mat = mabiki(chart_mat, 4) #4週ごと
	date_mat = mabiki(date_mat, 4) #4週ごと
	chart_mat = fill_deficit(chart_mat)

	return stock_list, chart_mat, date_mat


#時系列ポートフォリオリストを入れると、リターンを返す
def calcReturn(stock_portfolio, momMat):
	col = stock_portfolio.shape[0]
	row = stock_portfolio.shape[1]
	sim_inv = np.zeros(col)
	sim_inv[0] = 100
	for i in range(1,col):
		temp = 0
		for j in range(row):
			if stock_portfolio[i][j] != -1:
				temp = temp + momMat[i][int(stock_portfolio[i][j])]
			else:
				temp = temp + 1 #キャッシュを選択(-1)
		temp = temp/row
		sim_inv[i] = sim_inv[i-1] * temp
	return sim_inv


#一つのアセットだけで投資
def singleInvestSim(momMat, bsMat, no, startMonth):
	colSize = momMat.shape[0]
	simInv = np.zeros((colSize ,1))
	simInv[0][0] = 100
	for i in range(1, startMonth):
		simInv[i][0] = simInv[i-1][0] * momMat[i][no]
	for i in range(startMonth, colSize):
		if bsMat[i-1][no] > 0:
			simInv[i][0] = simInv[i-1][0] * momMat[i][no]
		else:
			simInv[i][0] = simInv[i-1][0]
	return simInv


#自作loss関数
def custom_loss(ytrue, ypred):
	#Kerasのバックエンドで記述しなければならない
	#https://keras.io/ja/backend/
	#ypredとytrueの正負が異なったとき、両者の二条誤差
	#ypredとytureの正負が一致したときは、誤差0
	return (-1)*K.minimum(K.sign(ytrue*ypred),0) * (ytrue-ypred)**2


#各月(mpara)リターンごとに重み付けした指数を返す
def mixMonthWeightReturn(chart_mat, mPara1, mPara2, mPara3, w1, w2, w3):
	c_size = chart_mat.shape[0] #行数
	r_size = chart_mat.shape[1] #列数
	ndarray = np.zeros(chart_mat.shape)
	for i in range(12,c_size):
		for j in range(r_size):
			"""
			ndarray[i][j] = chart_mat[i][j]* \
							( w1/(chart_mat[i-mPara1-1][j]+chart_mat[i-mPara1][j]+chart_mat[i-mPara1+1][j])*3 \
							+ w2/(chart_mat[i-mPara2-1][j]+chart_mat[i-mPara2][j]+chart_mat[i-mPara2+1][j])*3 \
							+ w3/(chart_mat[i-mPara3-1][j]+chart_mat[i-mPara3][j]+chart_mat[i-mPara3+1][j])*3 )
			"""
			#"""
			ndarray[i][j] = chart_mat[i][j]* \
							( w1/(chart_mat[i-mPara1-1][j]/2+chart_mat[i-mPara1][j]+chart_mat[i-mPara1+1][j]/2)*2 \
							+ w2/(chart_mat[i-mPara2-1][j]/2+chart_mat[i-mPara2][j]+chart_mat[i-mPara2+1][j]/2)*2 \
							+ w3/(chart_mat[i-mPara3-1][j]/2+chart_mat[i-mPara3][j]+chart_mat[i-mPara3+1][j]/2)*2 )
			#"""
			"""
			ndarray[i][j] = chart_mat[i][j]/chart_mat[i-mPara1][j]*w1 \
							+ chart_mat[i][j]/chart_mat[i-mPara2][j]*w2 \
							+ chart_mat[i][j]/chart_mat[i-mPara3][j]*w3
			"""
			if ndarray[i][j] == w1+w2+w3:#上場前だとこれになるので、0にしてしまう
				ndarray[i][j] = 0
	return ndarray


def return_daily_rank(chart_day_mat, dPara1, dPara2, dPara3, w1, w2, w3, drop_from_high_mat, rise_from_low_mat):
	col = chart_day_mat.shape[0]
	row = chart_day_mat.shape[1]
	rank_mat = np.zeros(chart_day_mat.shape)
	offset = 20
	for i in range(dPara3+offset, col):
		for j in range(row):
			if chart_day_mat[i-dPara3-offset][j]==0:
				rank_mat[i][j]=0
			else:
				#val = chart_day_mat[i][j] * (chart_day_mat[i][j]+2*chart_day_mat[i-offset][j]) / (3*chart_day_mat[i-offset][j])
				#rank_mat[i][j] = val * \
				rank_mat[i][j] = chart_day_mat[i][j]* \
							( w1/(chart_day_mat[i-dPara1-offset][j]/2+chart_day_mat[i-dPara1][j]+chart_day_mat[i-dPara1+offset][j]/2)*2 \
							+ w2/(chart_day_mat[i-dPara2-offset][j]/2+chart_day_mat[i-dPara2][j]+chart_day_mat[i-dPara2+offset][j]/2)*2 \
							+ w3/(chart_day_mat[i-dPara3-offset][j]/2+chart_day_mat[i-dPara3][j]+chart_day_mat[i-dPara3+offset][j]/2)*2 )
			rank_mat[i][j] *= 1 + (1-drop_from_high_mat[i][j])/2
			#rank_mat[i][j] *= 1 - (rise_from_low_mat[i][j]-1)/10
	return rank_mat

#time期間中高値からの下落をマップした行列
def return_drop_from_high(chart_day_mat, time):
	col = chart_day_mat.shape[0]
	row = chart_day_mat.shape[1]
	drop_mat = np.full((col,row), 0.0)
	for j in range(row):
		for i in range(time, col):
			max = np.amax(chart_day_mat[i-time:i,j])
			if max != 0:
				drop_mat[i][j] = float(chart_day_mat[i][j])/float(max)
			else:
				drop_mat[i][j] = 0.0
	return drop_mat


#time期間中安値からの上昇をマップした行列
def return_rise_from_low(chart_day_mat, time):
	col = chart_day_mat.shape[0]
	row = chart_day_mat.shape[1]
	rise_mat = np.full((col,row), 0.0)
	for j in range(row):
		for i in range(time, col):
			min = np.amin(chart_day_mat[i-time:i,j])
			if min != 0:
				rise_mat[i][j] = float(chart_day_mat[i][j])/float(min)
			else:
				rise_mat[i][j] = 0.0
	return rise_mat


#単純移動平均を返す
def returnSMA(originMat, num):
	#2次元の場合
	if originMat.ndim == 2:
		#originMat = 元行列，num = 移動平均パラメータ
		#元行列の単純移動平均行列を返す
		c_size = originMat.shape[0] #行数
		r_size = originMat.shape[1] #列数
		#maMat = np.ones(originMat.shape)
		b = np.ones((num,1))/num
		#print(b)
		maMat=signal.convolve2d(originMat, b, mode='valid')#移動平均
		defmat = np.zeros((num-1, r_size)) #0埋め
		maMat = np.vstack((defmat, maMat))
		return maMat
	#1次元の場合
	elif originMat.ndim==1:
		#print("1dim")
		c_size = originMat.shape[0]
		b = np.ones((num))/num
		maMat = signal.convolve(originMat, b, mode='valid')
		defMat = np.full(num-1, maMat[0])
		maMat = np.concatenate([defMat, maMat])
		#print(maMat)
		return maMat


def returnBorS(chart_mat, mov_ave_mat, mom_mat, threshold):
	BorS = np.zeros(chart_mat.shape)
	col = chart_mat.shape[0]
	row = chart_mat.shape[1]
	adj_chart_mat = np.ones((col,row))
	for i in range(col):
		for j in range(row):
			if mom_mat[i][j]>1:
				adj_chart_mat[i][j] = (0.5)*chart_mat[i][j] + (0.5)*chart_mat[i][j]*mom_mat[i][j]
			else:
				adj_chart_mat[i][j] = (0)*chart_mat[i][j] + (1.0)*chart_mat[i][j]*mom_mat[i][j]
	for i in range(col):
		for j in range(row):
			if mov_ave_mat[i][j] != 0:
				comp = adj_chart_mat[i][j] / mov_ave_mat[i][j]
				if comp > threshold:
					BorS[i][j] = 1
				else:
					BorS[i][j] = -1
	return(BorS)


def daily_return_bors(chart_mat, mov_ave_mat, threshold):
	BorS = np.zeros(chart_mat.shape)
	col = chart_mat.shape[0]
	row = chart_mat.shape[1]
	adj_chart_mat = np.ones((col,row))
	for i in range(col):
		for j in range(row):
			if chart_mat[i-20][j]==0:
				adj_chart_mat[i][j] = chart_mat[i][j]
			elif (chart_mat[i][j]/chart_mat[i-20][j])>1:
				adj_chart_mat[i][j] = (0.5)*chart_mat[i][j] + (0.5)*chart_mat[i][j]*(chart_mat[i][j]/chart_mat[i-20][j])
			else:
				adj_chart_mat[i][j] = (0)*chart_mat[i][j] + (1.0)*chart_mat[i][j]*(chart_mat[i][j]/chart_mat[i-20][j])
	for i in range(col):
		for j in range(row):
			if mov_ave_mat[i][j] != 0:
				comp = adj_chart_mat[i][j] / mov_ave_mat[i][j]
				if comp > threshold:
					BorS[i][j] = 1
				else:
					BorS[i][j] = -1
	return(BorS)


def returnSR(sim, mean):
	sim_momlist = np.zeros(len(sim)-1)
	mean_momlist = np.zeros(len(mean)-1)
	for i in range(len(sim_momlist)):
		sim_momlist[i] = (sim[i+1]/sim[i])*100-100
		mean_momlist[i] = (mean[i+1]/mean[i])*100-100
	excess_momlist = sim_momlist - mean_momlist
	#print(excess_momlist)
	sd = np.std(sim_momlist)
	if 1:
		print("sim平均リターン: %.2f%%" %np.mean(sim_momlist))
		#print("全銘柄平均リターン: %.2f%%" %np.mean(mean_momlist))
		print("平均超過リターン: %.2f%%" %np.mean(excess_momlist))
		print("標準偏差: %.2f%%" %sd)
	return(np.mean(excess_momlist)/sd)


def printRank(rankMat, stock_list, num):
	for i in range(len(rankMat)):
		for j in range(num):
			print(stock_list[rankMat[i][j]], end=' ')
		print()


def printRankMom(rankMat, momMat, stock_list):
	print("===  RANK Result  ===")
	row = rankMat.shape[1]
	count = 0
	for j in range(row):
		if j<9:
			print('0' + str(j+1), end=' ')
		else:
			print(str(j+1), end=' ')
		print("{:<8}".format(stock_list[rankMat[-2][j]][:6]), end='')#7文字目まで
		seiseki = round(momMat[-1][rankMat[-2][j]]*100-100, 2)
		if seiseki > 0:
			res = "+" + str(seiseki)
		else:
			res = str(seiseki)
		print('[' + res + ']', end=' ')
		count += 1
		if count==10:
			print()
			count=0
	print()


def print_latest_rank(rankMat, stock_list, chart_day_mat):
	print("===  Latest RANK  ===")
	row = rankMat.shape[1]
	count = 0
	for j in range(row):
		if j<9:
			print(' 0' + str(j+1), end=' ')
		elif j>=9 and j<99:
			print(' ' + str(j+1), end=' ')
		else:
			print(str(j+1), end=' ')
		print("{:<4}".format(stock_list[rankMat[-1][j]][:4]), end=' ')#7文字目まで
		print("[", end="")
		mom = chart_day_mat[-1][rankMat[-1][j]] / chart_day_mat[-1-20][rankMat[-1][j]] *100-100
		print( "{:>5.01f}%".format(mom), end='')
		print("] ",end="")
		count += 1
		if count==10:
			print()
			count = 0
	print()


def print_last_rank(rankMat, stock_list, chart_day_mat):
	print("===  Last RANK  ===")
	row = rankMat.shape[1]
	count = 0
	for j in range(row):
		if j<9:
			print(' 0' + str(j+1), end=' ')
		elif j>=9 and j<99:
			print(' ' + str(j+1), end=' ')
		else:
			print(str(j+1), end=' ')
		print("{:<4}".format(stock_list[rankMat[-2][j]][:4]), end=' ')#7文字目まで
		print("[", end="")
		mom = chart_day_mat[-1][rankMat[-2][j]] / chart_day_mat[-1-20][rankMat[-2][j]] *100-100
		print( "{:>5.01f}%".format(mom), end='')
		print("] ",end="")
		count += 1
		if count==10:
			print()
			count = 0
	print()


def printBSRank(rankMat, bsMat, stock_list, num):
	for i in range(len(rankMat)):
		for j in range(num):
			print(stock_list[rankMat[i][j]], end=' ')
			if bsMat[i][rankMat[i][j]]>0:
				print("b", end=' ')
			else:
				print("s", end=' ')
		print()


#開始月からのデータの無い(データが足りていない)列にたいして、保管する
#終了月を合わせるようにずれして、前データをすべて99999(or初期値)で埋める
def fill_deficit(chart_mat):
	col = chart_mat.shape[0]
	row = chart_mat.shape[1]
	for j in range(row):
		i = 0
		while i < col:
			if np.isnan(chart_mat[i][j])==False:
				fy = chart_mat[i][j]
				break
			i += 1
		chart_mat[:,j] = chart_mat[:,j]/fy*100
	for j in range(row):
		if np.isnan(chart_mat[0][j])==True:
			for i in range(col):
				if np.isnan(chart_mat[i][j])==True:
					chart_mat[i][j]=FILL_DEFICIT
	return(chart_mat)

def fill_deficit2(chart_mat):
	col = chart_mat.shape[0]
	row = chart_mat.shape[1]
	"""
	for j in range(row):
		i = 0
		while i < col:
			if chart_mat[i][j]!=0:
				fy = chart_mat[i][j]
				break
			i += 1
		chart_mat[:,j] = chart_mat[:,j]/fy*100
	"""
	for j in range(row):
		if chart_mat[0][j]==0:
			for i in range(col):
				if chart_mat[i][j]==0:
					chart_mat[i][j]=FILL_DEFICIT
	return(chart_mat)


def mabiki(mat, n):
	#終了点固定でまびき
	if type(mat) is np.ndarray:
		col = mat.shape[0]
		row = mat.shape[1]
		q, mod = divmod(col, n)
		if mod <= n-2:
			mat = np.delete(mat, np.s_[0:mod], 0)
		for i in range(q):
			mat = np.delete(mat, np.s_[-n-i:-1-i], 0)
		#print(fafda)
		return(mat)
	elif type(mat) is list:
		l = len(mat)
		q, mod = divmod(l, n)
		if mod <= n-2:
			mat = np.delete(mat, np.s_[0:mod], 0)
		for i in range(q):
			mat = np.delete(mat, np.s_[-n-i:-1-i], 0)
		return(mat)

def mabiki_from_start(mat, n):
	#開始点固定でまびき
	if type(mat) is np.ndarray:
		col = mat.shape[0]
		row = mat.shape[1]
		q, mod = divmod(col, n)
		#print("amari is ", end='')
		#print(mod)
		#print("col is ", end="")
		#print(col)
		if mod <= n-2 and mod > 0:
			mat = np.delete(mat, np.s_[-mod:], 0)
		#print("col is ", end="")
		#print(mat.shape[0])
		for i in range(q):
			mat = np.delete(mat, np.s_[-n-i:-1-i], 0)
		return(mat)
	elif type(mat) is list:
		l = len(mat)
		q, mod = divmod(l, n)
		if mod <= n-2:
			mat = np.delete(mat, np.s_[0:mod], 0)
		for i in range(q):
			mat = np.delete(mat, np.s_[-n-i:-1-i], 0)
		return(mat)



def evalRank(rankIndex, momMat):
	col = rankIndex.shape[0]
	row = rankIndex.shape[1]
	rankMom = np.zeros((1,row))
	#print(momMat)
	for i in range(13,(col-1)):
		for j in range(row):
			rankMom[0][j] = rankMom[0][j] + (momMat[i+1][rankIndex[i][j]]*100-100)
		#print(rankMom)
	rankMom = rankMom / col
	print(rankMom)


def maxDD(result):
	size = result.size
	max = 0
	temp_dd = 0
	mdd = 0
	for i in range(size):
		if max < result[i]:
			max = result[i]
		temp_dd = 100-result[i]/max*100
		if temp_dd > mdd:
			mdd = temp_dd
	return mdd


#前月比mat
def return_mom_mat(chart_mat):
	colSize = chart_mat.shape[0]
	rowSize = chart_mat.shape[1]
	mom_mat = np.ones((colSize,rowSize))
	for i in range(1,colSize):
		for j in range(rowSize):
			mom_mat[i][j] = chart_mat[i][j]/chart_mat[i-1][j];
	#np.set_printoptions(precision=2)
	return mom_mat


#全銘柄に平均投資
#初期からデータのない銘柄もあるので、それを排除
#fillDeficitでそのような銘柄は-1でデータを埋めている

def meanInv(chart_day_mat, startday):
	sd = startday
	mean = np.ones(chart_day_mat.shape[0])
	for i in range(sd):
		mean[i] = 100
	for i in range(sd, chart_day_mat.shape[0]):
		mean_dod = 0
		zero_count = 0
		for j in range(chart_day_mat.shape[1]):
			if chart_day_mat[i-1][j]!=0:
				mean_dod += chart_day_mat[i][j]/chart_day_mat[i-1][j]
			else:
				zero_count += 1
		mean_dod = mean_dod/(chart_day_mat.shape[1]-zero_count)
		mean[i] = mean[i-1] * mean_dod
	return mean

#複数のチャートをmatplotで表示 引数可変
def show_plots(*chart):
	plt.yscale("log")
	for i in range(len(chart)):
		plt.plot(chart[i])
	plt.xticks(rotation=90)
	plt.legend()
	plt.show()

def show_plot(chart):
	plt.yscale("log")
	plt.plot(chart)
	plt.xticks(rotation=90)
	plt.legend()
	plt.show()

def show_plot_sp500_vs_inv(result, chart_mat_SP500):
	plt.yscale("log")
	#labels = np.arange('1997-07', '2019-01', dtype='datetime64[M]')
	#ticks=12
	#plt.xticks(labels[::ticks])
	#plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m'))
	plt.plot(result, label='simInv')
	#plt.plot(labels, sp500Inv, label='sp500Inv')
	#plt.plot(chart_mat_facebook/chart_mat_facebook[0]*100, label='facebook')
	plt.plot(chart_mat_SP500/chart_mat_SP500[0]*100, label='S&P500')
	#smov_ave_mat_SP500 = mfiv.returnSMA(chart_mat_SP500, 12)
	#plt.plot(smov_ave_mat_SP500/chart_mat_SP500[0]*100, label='S&P500_sma-1y')
	plt.xticks(rotation=90)
	plt.legend()
	plt.show()

def show_plot_sp500_vs_sma(chart_mat_SP500, month):
	plt.yscale("log")
	plt.plot(chart_mat_SP500/chart_mat_SP500[0]*100, label='S&P500')
	smov_ave_mat_SP500 = returnSMA(chart_mat_SP500, month)
	plt.plot(smov_ave_mat_SP500/chart_mat_SP500[0]*100, label='S&P500_sma-1y')
	plt.xticks(rotation=90)
	plt.legend()
	plt.show()


def random_delete_stock(stock_lsit, chart_mat):
	trow = chart_mat.shape[1]
	dellist = []
	for i in range(trow):
		if rnd.random() > 0.6 and i!=76:
			dellist.append(i)
	chart_mat = np.delete(chart_mat, dellist, 1)
	delmethod = lambda items, indexes: [item for index, item in enumerate(items) if index not in indexes]
	stock_list = delmethod(stock_list, dellist)
	return stock_list, chart_mat


def sim_bs_th_exam(i, j, chart_mat, mov_ave_mat, mom_mat, rankIndex, rankPointMat, rank_buy_th, rank_sell_th, start_month, stock_list):
	for i in range(i):
		for j in range(j):
			thresh = 0.5 + i * 0.1
			print("buy=%.2f" %thresh, end=" ")
			bs_buy_mat = returnBorS(chart_mat, mov_ave_mat, mom_mat, thresh)
			thresh = 0.5 + j * 0.05
			print("sell=%.2f" %thresh, end=" ")
			bs_sell_mat = returnBorS(chart_mat, mov_ave_mat, mom_mat, thresh)
			SIM = IS.InvSim(mom_mat, rankIndex, rankPointMat, bs_buy_mat, bs_sell_mat, rank_buy_th, rank_sell_th, start_month, stock_list)
			SIM.set_hold_num(5)
			SIM.sim_run(0)
			print("result=%.1f" %SIM.result[-1], end=" ")
			#s_ratio = returnSR( np.delete(SIM.result, slice(start_month), 0), mean_inv_sim_mat)
			#print("sharpeRatio = %.3f" %s_ratio)
			max_draw_down = maxDD(SIM.result)
			print("Max Draw Down: %.1f%%" %max_draw_down)


def sim_rank_th_exam(i, j, chart_mat, mov_ave_mat, mom_mat, rankIndex, rankPointMat, bs_buy_mat, bs_sell_mat, start_month, stock_list):
	for i in range(i):
		for j in range(j):
			rank_buy_th = 1.0 + 0.05*i
			print("buy=%.2f" %rank_buy_th, end=" ")
			rank_sell_th = 1.0 + 0.05*j
			print("sell=%.2f" %rank_sell_th, end=" ")
			SIM = IS.InvSim(mom_mat, rankIndex, rankPointMat, bs_buy_mat, bs_sell_mat, rank_buy_th, rank_sell_th, start_month, stock_list)
			SIM.set_hold_num(5)
			SIM.sim_run(0)
			print("result=%.1f" %SIM.result[-1], end=" ")
			#s_ratio = returnSR( np.delete(SIM.result, slice(start_month), 0), mean_inv_sim_mat)
			#print("sharpeRatio = %.3f" %s_ratio)
			max_draw_down = maxDD(SIM.result)
			print("Max Draw Down: %.1f%%" %max_draw_down)

#外れ値を除外した標準化
def standard_ex_outlier(x, out):
	#xはn行1列を想定
	mean = 0
	temp_sum = 0
	count = 0
	list = []
	for i in range(x.shape[0]):
		if x[i,0]!=out:
			temp_sum = temp_sum + x[i,0]
			count = count + 1
			list.append(x[i,0])
	mean = temp_sum / count
	std = stdev(list)
	for j in range(x.shape[0]):
		if x[j,0]!=out:
			x[j,0] = (x[j,0]-mean)/std
	return x

"""
def invest_simlation(momMat, rankIndex, bsMat, startMonth, topnum):
	#   chart_mat: チャートデータ
	#   rankIndex: chart_matをランキングしたインデックス行列
	#		1列目1位，2列目2位，...
	#   bsMat: chart_mat各要素ごとに買付基準を満たしているか 1 or -1
	#   startMonth: シミュレーション開始月
	colSize = momMat.shape[0] #行数
	#simInv = np.zeros((colSize ,1));
	simInv = np.zeros(colSize)
	initialData = 100;
	#topnum = 8;
	print("the num of stock is " + str(topnum))

	for i in range(startMonth):
		#simInv[i][0] = initialData
		simInv[i] = initialData
	accordNum =0;
	accordMat = np.ones((1,3))

	for i in range(startMonth, colSize):
		top_coef = 0
		for j in range(topnum):
			if bsMat[i-1][rankIndex[i-1][j]] > 0:#jでなくj+1、つまり1位ははずす
				top_coef = top_coef + momMat[i][rankIndex[i-1][j]]#jでなくj+1、つまり1位ははずす
				accordNum = accordNum + 1
			else:
				top_coef = top_coef + 1
				accordMat[0][0] = 0;
		#simInv[i][0] = simInv[i-1][0] * (top_coef/topnum)
		simInv[i] = simInv[i-1] * (top_coef/topnum)

	return simInv
"""


'''
function [ maMat ] = returnMovingAverage( originalMat, movePara )
%MOVE この関数の概要をここに記述
%  original = 元行列，movePara = 移動平均パラメータ
%  元行列の移動平均行列を返す

sz = size(originalMat);
maMat = ones(sz);

tempSum=0;
for i = movePara:max(sz)
	for j = 1:min(sz)
		for k=0:movePara-1
			tempSum = tempSum + originalMat(i-movePara+k+1,j);
		end
		maMat(i,j) = tempSum/movePara;
		tempSum=0;
	end
end
end
'''


"""
#極力銘柄入れ替えを減らす
#n銘柄を保有、毎月最もrankPointの低い銘柄をチェンジする
#チェンジ先は、保有していない銘柄の中で最もrankPointの高い銘柄
#しかし、その銘柄が全体ランク1位の場合は次点の銘柄を選択する(ランク1位は危険ぽいので)
#bsMatが満たされない場合はcashを保有する(-1番)
#cashが頻繁に選ばれる?
#todo:1銘柄偏り対策、ただし売買回数は減らしたい
#最も比率の高い銘柄と低い銘柄の比が2を超えた場合、２つを平均化する
#早い段階でalibabaが現れるバグ
#プログラムのミスして気づいたけど一番儲かっていないのを交換する方法が儲かる？
def invest_simlation_2(momMat, rankIndex, rankPoint, bsMatbuy, bsMatsell, rankbuy, ranksell, startMonth, hold_num, stock_list):
	#   chart_mat: チャートデータ
	#   rankIndex: chart_matをランキングしたインデックス行列
	#		1列目1位，2列目2位，...
	#   bsMat: chart_mat各要素ごとに買付基準を満たしているか 1 or -1
	#   startMonth: シミュレーション開始月
	#   num_hold_stock: 保有する銘柄数
	col = momMat.shape[0] #行数
	row = momMat.shape[1]
	hold_stock_mat = np.full((col+1, hold_num),-1)#時系列ポートフォリオリスト#次月の予測含めるのでcol+1
	ptfl = np.zeros((3, hold_num))#0行目:stockナンバー,#1行目:損益,#2行目:rank指数
	#print("the num of stock is " + str(hold_num))
	result = np.ones(col)
	initial = 20

	#開始月は普通にランク上位をポートフォリオに
	count = 0
	j = 0
	while j < row:
		if bsMatbuy[startMonth-1][rankIndex[startMonth-1][j]] > 0:
			hold_stock_mat[startMonth][count] = rankIndex[startMonth-1][j]
		count += 1
		if count == hold_num:
			break
		j += 1
	for j in range(hold_num):
		ptfl[0][j] = hold_stock_mat[startMonth][j]
		ptfl[1][j] = initial #初期値
		ptfl[2][j] = rankPoint[startMonth][int(ptfl[0][j])]

	ptfl = ptfl[:, np.argsort(ptfl[2])] #rankPointの最下位を先頭にソートする

	for i in range(startMonth+1):
		result[i] = 100

	#全ループ
	for i in range(startMonth+1, col):
		#
		for j in range(hold_num):
			ptfl[0][j] = hold_stock_mat[i][j]
			if ptfl[0][j]!=-1:
				#cash(-1)でない場合、前回の値*前月比
				ptfl[1][j] *= momMat[i][int(ptfl[0][j])]
				#rankPointを更新
				ptfl[2][j] = rankPoint[i][int(ptfl[0][j])]
			else:#キャッシュならrankPointは1.2
				ptfl[2][j] = 1.2
		result[i] = sum(ptfl[1][:])

		#保有額1位と最下位の比が2を超えている場合、是正
		ptfl = ptfl[:, np.argsort(ptfl[1])] #保有額でソートする
		if ptfl[1][-1]/ptfl[1][0] > 2:
			#print("hoge")
			temp = ptfl[1][0] + ptfl[1][-1]
			ptfl[1][0] = temp/2
			ptfl[1][-1] = temp/2
			print("averaging because of disparity")

		ptfl = ptfl[:, np.argsort(ptfl[2])] #rankPointの最下位を先頭にソートする

		#ナンバーを名前に変えてprint
		if 1:
			for k in range(hold_num):
				if ptfl[0][k]!=-1:
					print("{:<8}".format(stock_list[int(ptfl[0][k])][:8]), end=' ')#7文字目まで
				else:
					print("{:<8}".format("cash"), end=' ')#7文字目まで
					#print()
			print()
			print(ptfl[1:][:])

		#rankIndex行からすでにportfolioに含まれているナンバーを削除し、先頭(1位)を新たにポートフォリオ先頭に加える
		#~:否定演算子、in1d:一致するインデックスを返す #[1:]のところi.e.先頭値も候補
		ranklist = rankIndex[i][1:]#1位は回避
		ranklist = ranklist[~np.in1d(ranklist, ptfl[0][1:])]
		chflag = np.zeros(hold_num)#変更フラグ
		if i!=col:#最後は無し
			chflag[0] = 1#先頭は変更
			for m in range(1,hold_num):
				#cashであるか、ranksellを下回るか、bsMatsellが0なら
				if ptfl[0][m] == -1 or ptfl[2][m]<ranksell or bsMatsell[i][int(ptfl[0][m])]!=1:
					chflag[m] = 1
			count = 0
			for n in range(hold_num):
				#if n==0:
					#print("rankbuy:%.2f"%rankPoint[i][ranklist[count]], end=" ")
					#print("bsMatbuy:%.2f"%bsMatbuy[i][ranklist[count]])
				if chflag[n] == 1:
					#if (rankPoint[i][ranklist[count]]>rankbuy and bsMatbuy[i][ranklist[count]]==1) or n==0:
					flag = 0
					while count<row-hold_num:
						if rankPoint[i][ranklist[count]]>rankbuy and bsMatbuy[i][ranklist[count]]==1:
							#print(rankPoint[i][ranklist[count]])
							hold_stock_mat[i+1][n] = ranklist[count]
							count += 1
							flag = 1
							break
						count += 1
					if flag == 0:
						if ptfl[2][n]>=1.5:
							hold_stock_mat[i+1][n] = ptfl[0][n]
						else:
							hold_stock_mat[i+1][n] = -1
				else:
					hold_stock_mat[i+1][n] = ptfl[0][n]


			#ここにこれを入れるとバグ起きるけど、すごい良い成績になる→なにかのヒント？
			#保有額1位と最下位の比が2を超えている場合、是正
			#ptfl = ptfl[:, np.argsort(ptfl[1])]
			#if ptfl[1][-1]/ptfl[1][0] > 2:
			#	temp = ptfl[1][0] + ptfl[1][-1]
			#	ptfl[1][0] = temp/2
			#	ptfl[1][-1] = temp/2

	#次の予想
	if 1:
		print()
		print("*** NextMonth ***",end="")
		for k in range(hold_num):
			if hold_stock_mat[col][k]!=-1:
				print("{:<8}".format(stock_list[int(hold_stock_mat[col][k])][:8]), end='')#7文字目まで
				print("[" + str(hold_stock_mat[col][k]) + "]", end=' ')
			else:
				print("{:<8}".format("cash"), end=' ')#7文字目まで
				print()
	return result,hold_stock_mat
"""
