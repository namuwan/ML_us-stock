import pandas as pd
import numpy as np

import my_function_individual as mfiv

class Chart:
    """Contain Data"""
    def __init__(self, chart_df):
        self.df = chart_df
        self.stock_list = list(chart_df.columns.values)
        del self.stock_list[0] #delete first item "date"
        self.stock_list.insert(0, "cash")
        self.stock_list.insert(2, "inverse")
        self.stock_list.insert(3, "double-inv")
        self.date_array = chart_df.iloc[:,0:1].values.flatten()
        self.date_array = np.delete(self.date_array, 0) #先頭のnanを削除
        self.chart_day_mat = chart_df.values
        self.chart_week_mat = np.array([])
        self.week_index = []
        self.chart_month_mat = np.array([])
        self.month_index = []
        self.mom_mat = np.array([])
        self.col = 0
        self.row = 0
        self.date_length = 0
        #"""
        self.rank_point_mat = []
        self.rank_index = []
        self.mov_ave_mat = []
        self.bs_buy_mat = []
        self.bs_sell_mat = []
        #"""
        self.daily_rank_mat = []
        self.daily_rank_index = []
        self.daily_mov_ave_mat = []
        self.daily_bs_buy_mat = []
        self.daily_bs_sell_mat = []

        self.rank_buy_thresh = 0
        self.rank_sell_thresh = 0

        self.drop_from_high_mat = []
        self.rise_from_low_mat = []

        #ML用
        self.next_month_ex_return = []
        #self.n_day_ex_return = []

        self.tnx_mat = []
        self.volume_day_mat = []
        self.volume_rate_mat = []

        self.chart_day_change_mat = []

    def set_data(self):
        self._set_day_mat()
        cash_mat = np.full((self.chart_day_mat.shape[0], 1), 100) #現金
        self.chart_day_mat = np.concatenate([cash_mat, self.chart_day_mat], 1) #現金列を0行目に追加
        inverse_mat = np.full((self.chart_day_mat.shape[0]), 1000.0)
        d_inverse_mat =  np.full((self.chart_day_mat.shape[0]), 1000.0)
        for i in range(1, self.chart_day_mat.shape[0]):
            inverse_mat[i] = inverse_mat[i-1] * (2-(self.chart_day_mat[i][1]/self.chart_day_mat[i-1][1]))
            d_inverse_mat[i] = d_inverse_mat[i-1] * (3-2*(self.chart_day_mat[i][1]/self.chart_day_mat[i-1][1]))
        #print(d_inverse_mat)
        self.chart_day_mat = np.insert(self.chart_day_mat, 2, inverse_mat, axis=1)
        self.chart_day_mat = np.insert(self.chart_day_mat, 3, d_inverse_mat, axis=1)
        self.date_length = self.chart_day_mat.shape[0]
        #self._set_week_mat()
        #self._set_month_mat()
        #self._set_mom_mat()
        #self.col = self.chart_month_mat.shape[0]
        #self.row = self.chart_month_mat.shape[1]
        self._set_stock_name()
        if self.date_array.size != self.chart_day_mat.shape[0]:
        	print("DATA Length No Match in ChartData.py")
        	sys.exit(1)
        #self.drop_from_high_mat = mfiv.return_drop_from_high(self.chart_day_mat, 100)
        #self.rise_from_low_mat =  mfiv.return_rise_from_low(self.chart_day_mat, 100)

        #for ML
        self.set_next_month_ex_return()
        #self.set_n_day_ex_return(20)


    def set_volume_data(self, volume_df):
        self.volume_day_mat = volume_df.values
        self.volume_day_mat = np.delete(self.volume_day_mat, 0, axis=1)
        self.volume_day_mat = np.delete(self.volume_day_mat, 0, axis=0)
        self.volume_day_mat = self.volume_day_mat.astype(np.float32)
        #"""
        for j in range(self.volume_day_mat.shape[1]):
            first_data = -999
            listed = 0 #上場
            for i in range(self.volume_day_mat.shape[0]):
                if listed==0 and self.volume_day_mat[i][j]!=0:
                    listed = 1
                    first_data = self.volume_day_mat[i][j]
                    self.volume_day_mat[i][j] = 100
                if listed==1:
                    if self.volume_day_mat[i][j] != 0:
                        self.volume_day_mat[i][j] = self.volume_day_mat[i][j]/first_data*100
                    else:
                        self.volume_day_mat[i][j] = self.volume_day_mat[i-1][j] #出来高0の場合は前日と同じ値
        self.volume_rate_mat = np.full(self.volume_day_mat.shape, -999.0)
        #"""
        #print(self.volume_day_mat.shape)
        #print(self.volume_day_mat)
        #hoge

    def ret_RSI(self, volume_df):


    def _set_day_mat(self):
        self.chart_day_mat = np.delete(self.chart_day_mat, 0, axis=1)
        self.chart_day_mat = np.delete(self.chart_day_mat, 0, axis=0)
        self.chart_day_mat = self.chart_day_mat.astype(np.float32)

    #extract friday data only from df
    def _set_week_mat(self):
        temp_df = self.df.drop([0])
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        friday_list = []
        row_num, pre_dow = 0, 0
        for index, item in temp_df.iterrows():
            dow = item['date'].dayofweek #Monday is 0, Sunday is 6
            if dow < pre_dow:#週が明けたら
                friday_list.append(row_num-1) #row_num-1が先週末最後の曜日、多くの場合金曜日
            pre_dow = dow
            row_num += 1
        #データ最終日が金曜でなければ最終日をデータに追加
        if friday_list[-1] != temp_df.shape[0]-1:
            friday_list.append(temp_df.shape[0]-1)
        #chart_week_df = temp_df.loc[friday_list]
        #chart_week_mat = chart_week_df.values
        temp_np = temp_df.values
        chart_week_mat = temp_np[friday_list]
        chart_week_mat = np.delete(chart_week_mat, 0, axis=1) # delete first row (date)
        chart_week_mat = np.delete(chart_week_mat, 0, axis=0) # delete first column (ticker name)
        chart_week_mat = chart_week_mat.astype(np.float32)
        cash_mat = np.full((chart_week_mat.shape[0], 1), 100)
        chart_week_mat = np.concatenate([cash_mat, chart_week_mat], 1)
        self.chart_week_mat = chart_week_mat
        self.week_index = friday_list

    def _set_month_mat(self):
        self.chart_month_mat = mfiv.mabiki_from_start(self.chart_week_mat, 4)
        self.chart_month_mat = mfiv.fill_deficit2(self.chart_month_mat)
        self.month_index = mfiv.mabiki_from_start(self.week_index, 4)

    def _set_mom_mat(self):
        self.mom_mat = mfiv.return_mom_mat(self.chart_month_mat)

    def _set_stock_name(self):
        for i in range(self.row):
            self.stock_list[i] = self.stock_list[i].replace(" ", "_")
            self.stock_list[i] = self.stock_list[i].replace("&", "_")
            self.stock_list[i] = self.stock_list[i].replace("\'", "_")
            self.stock_list[i] = self.stock_list[i].replace("(", "_")
            self.stock_list[i] = self.stock_list[i].replace(")", "")
            #exec("chart_mat_" + stock_list[i] + " = chart_mat[:," + str(i) + "]")

    def set_rank_point_mat(self, mPara1, mPara2, mPara3, w1, w2, w3):
        self.rank_point_mat = mfiv.mixMonthWeightReturn(self.chart_month_mat, mPara1, mPara2, mPara3, w1, w2, w3)
        self.rank_index = np.argsort(self.rank_point_mat)[:,::-1]

    def set_daily_rank_mat(self, dPara1, dPara2, dPara3, w1, w2, w3):
        self.daily_rank_mat = mfiv.return_daily_rank(self.chart_day_mat, dPara1, dPara2, dPara3, w1, w2, w3, self.drop_from_high_mat, self.rise_from_low_mat)
        for i in range(self.date_array.size):#インバース系はrankpointを0にする
            self.daily_rank_mat[i][2]=0
            self.daily_rank_mat[i][3]=0
        self.daily_rank_index = np.argsort(self.daily_rank_mat)[:,::-1]

    def set_mov_ave_mat(self, mov_ave_para):
        self.mov_ave_mat = mfiv.returnSMA(self.chart_month_mat, mov_ave_para)

    def set_daily_mov_ave_mat(self, mov_ave_para):
        self.daily_mov_ave_mat = mfiv.returnSMA(self.chart_day_mat, mov_ave_para)

    def set_bs_mat(self, buy_thresh, sell_thresh):
        self.bs_buy_mat = mfiv.returnBorS(self.chart_month_mat, self.mov_ave_mat, self.mom_mat, buy_thresh)
        self.bs_sell_mat = mfiv.returnBorS(self.chart_month_mat, self.mov_ave_mat, self.mom_mat, sell_thresh)

    def set_daily_bs_mat(self, buy_thresh, sell_thresh):
        self.daily_bs_buy_mat = mfiv.daily_return_bors(self.chart_day_mat, self.daily_mov_ave_mat, buy_thresh)
        self.daily_bs_sell_mat = mfiv.daily_return_bors(self.chart_day_mat, self.daily_mov_ave_mat, sell_thresh)

    def set_rank_thresh(self, buy, sell):
        self.rank_buy_thresh = buy
        self.rank_sell_thresh = sell

    # NASDAQをベンチマークとしたときの超過リターン
    def set_next_month_ex_return(self):
        self.next_month_ex_return = np.full((self.chart_day_mat.shape), -999.0)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(col-20):
            for j in range(row):
                if self.chart_day_mat[i][j] != 0:
                    self.next_month_ex_return[i][j] = (self.chart_day_mat[i+20][j]/self.chart_day_mat[i][j])*100-100
                    #bench_mark = self.chart_day_mat[i+20][1]/self.chart_day_mat[i][1]*100-100 #SP500 return
                    bench_mark = self.nasdaq_day_mat[i+20][0]/self.nasdaq_day_mat[i][0]*100-100 #SP500 return
                    self.next_month_ex_return[i][j] = self.next_month_ex_return[i][j] - bench_mark
                else:
                    self.next_month_ex_return[i][j] = -999.0

    # n日前からの超過リターン
    def ret_n_day_ex_return(self, n):
        n_day_ex_return = np.full((self.chart_day_mat.shape), -999.0)
        #self.n_day_ex_return = np.full((self.chart_day_mat.shape), -999)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(n, col):
            for j in range(row):
                if self.chart_day_mat[i-n][j] != 0:
                    n_day_ex_return[i][j] = (self.chart_day_mat[i][j]/self.chart_day_mat[i-n][j])*100-100
                    #bench_mark = self.chart_day_mat[i][1]/self.chart_day_mat[i-n][1]*100-100 #SP500 return
                    bench_mark = self.nasdaq_day_mat[i][0]/self.nasdaq_day_mat[i-n][0]*100-100 #SP500 return
                    n_day_ex_return[i][j] = n_day_ex_return[i][j] - bench_mark
                else:
                    n_day_ex_return[i][j] = -999.0 #欠損値は-999
        n_day_ex_return = np.delete(n_day_ex_return, [0,1,2,3], 1) #SPY, cashなどを除く
        n_day_ex_return = n_day_ex_return.reshape(-1,1)
        return n_day_ex_return

    # n日前からの超過リターン(n日前の値は平均化)
    def ret_n_day_ex_mean_return(self, n):
        n_day_ex_mean_return = np.full((self.chart_day_mat.shape), -999.0)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        mean_range = 20
        for i in range(n+mean_range, col):
            for j in range(row):
                if self.chart_day_mat[i-n-mean_range][j] != 0:
                    n_day_ago_average = np.mean(self.chart_day_mat[i-n-mean_range:i-n+mean_range, j])
                    #n_day_ago_average = (1/2) * ( self.chart_day_mat[i-n-mean_range, j]*(1/2)+self.chart_day_mat[i-n,j]+self.chart_day_mat[i-n+mean_range,j]*(1/2) )
                    n_day_ex_mean_return[i][j] = (self.chart_day_mat[i][j]/n_day_ago_average)*100-100
                    bench_mark = self.nasdaq_day_mat[i][0]/self.nasdaq_day_mat[i-n][0]*100-100 #SP500 return
                    n_day_ex_mean_return[i][j] = n_day_ex_mean_return[i][j] - bench_mark
                else:
                    n_day_ex_mean_return[i][j] = -999.0 #欠損値は-999
        n_day_ex_mean_return = np.delete(n_day_ex_mean_return, [0,1,2,3], 1) #SPY, cashなどを除く
        n_day_ex_mean_return = n_day_ex_mean_return.reshape(-1,1)
        return n_day_ex_mean_return

    #過去n日の高値からの下落率
    def ret_n_day_drop(self, n):
        n_day_drop = np.full((self.chart_day_mat.shape), -999.0)
        #print(self.chart_day_mat[0:20,:])
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(n, col):
            for j in range(row):
                #print(self.chart_day_mat[i-n:i,j].max())
                temp_max = self.chart_day_mat[i-n:i,j].max()
                if temp_max!=-999 and temp_max!=0:
                    n_day_drop[i][j] = 100.0 - self.chart_day_mat[i][j]/temp_max*100
                else:
                    n_day_drop[i][j] = -999.0
        n_day_drop = np.delete(n_day_drop, [0,1,2,3], 1) #SPY, cashなどを除く
        n_day_drop = n_day_drop.reshape(-1,1)
        return n_day_drop

    #過去n日の高値から何日経っているか
    def ret_n_day_under(self, n):
        n_day_under = np.full((self.chart_day_mat.shape), -999)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(n, col):
            for j in range(row):
                #if self.chart_day_mat[i-n][j]!=self.chart_day_mat[i][j]:
                if self.chart_day_mat[i-n][j]!=0:
                    argmax = np.argmax(self.chart_day_mat[i-n:i,j])
                    n_day_under[i][j] = n-argmax
                else:
                    n_day_under[i][j] = -999
        n_day_under = np.delete(n_day_under, [0,1,2,3], 1) #SPY, cashなどを除く
        #print(n_day_under)
        n_day_under = n_day_under.reshape(-1,1)
        return n_day_under

    #新高値からn日以内にいるか
    def ret_new_highs(self, n):
        during = 220
        new_highs = np.full((self.chart_day_mat.shape), -999)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(220, col):
            for j in range(row):
                #if self.chart_day_mat[i-during][j]!=self.chart_day_mat[i][j]:
                if self.chart_day_mat[i-during][j]!=0:
                    argmax = np.argmax(self.chart_day_mat[i-during:i,j])
                    if during-argmax < n:
                        new_highs[i][j] = 1
                    else:
                        new_highs[i][j] = 0
                else:
                    new_highs[i][j] = -999
        new_highs = np.delete(new_highs, [0,1,2,3], 1) #SPY, cashなどを除く
        new_highs = new_highs.reshape(-1,1)
        return new_highs

    #過去n日の最低値からの上昇率
    def ret_n_day_bottom_rise(self, n):
        n_day_bottom_rise = np.full((self.chart_day_mat.shape), -999.0)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(n, col):
            for j in range(row):
                temp_min = self.chart_day_mat[i-n:i,j].min()
                if temp_min!=-999 and temp_min!=0:
                    n_day_bottom_rise[i][j] = self.chart_day_mat[i][j]/temp_min*100-100
                else:
                    n_day_bottom_rise[i][j] = -999.0
        n_day_bottom_rise = np.delete(n_day_bottom_rise, [0,1,2,3], 1) #SPY, cashなどを除く
        n_day_bottom_rise = n_day_bottom_rise.reshape(-1,1)
        return n_day_bottom_rise

    #n日移動平均線からの乖離率(divergence)
    def ret_div_sma(self, n):
        div_sma = np.full((self.chart_day_mat.shape), -999.0)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(n, col):
            for j in range(row):
                if self.chart_day_mat[i-n][j]!=0:
                    sma_value = np.mean(self.chart_day_mat[i-n:i,j])
                    div_sma[i][j] = self.chart_day_mat[i][j]/sma_value*100-100
        div_sma = np.delete(div_sma, [0,1,2,3], 1) #SPY, cashなどを除く
        #print(div_sma)
        div_sma = div_sma.reshape(-1,1)
        return div_sma

    #n日移動平均線の上or下
    def ret_above_sma(self, n):
        abv_sma = np.full((self.chart_day_mat.shape), -999)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(n, col):
            for j in range(row):
                if self.chart_day_mat[i-n][j]!=0:
                    sma_value = np.mean(self.chart_day_mat[i-n:i,j])
                    if self.chart_day_mat[i][j] > sma_value:
                        abv_sma[i][j] = 1
                    else:
                        abv_sma[i][j] = 0
        abv_sma = np.delete(abv_sma, [0,1,2,3], 1) #SPY, cashなどを除く
        abv_sma = abv_sma.reshape(-1,1)
        return abv_sma

    """
    def ret_n_days_return(self, n):
        n_day_return = np.full((self.chart_day_mat.shape), -999.0)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(n, col):
            for j in range(row):
                n_day_return[i][j] = chart_day_mat/contain_month_mat
    """

    #220日移動平均線より上 かつ n日移動平均線より上
    def ret_gld_cross(self, n):
        gld_cross = np.full((self.chart_day_mat.shape), -999)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(n, col):
            for j in range(row):
                if self.chart_day_mat[i-220][j]!=0:
                    sma_220_value = np.mean(self.chart_day_mat[i-220:i,j])
                    sma_value = np.mean(self.chart_day_mat[i-n:i,j])
                    if self.chart_day_mat[i][j]>sma_220_value and self.chart_day_mat[i][j]>sma_value:
                        gld_cross[i][j] = 1
                    else:
                        gld_cross[i][j] = 0
        gld_cross = np.delete(gld_cross, [0,1,2,3], 1) #SPY, cashなどを除く
        gld_cross = gld_cross.reshape(-1,1)
        return gld_cross

    #n日のボラティリティ
    def ret_volatility(self, n):
        vola_mat = np.full((self.chart_day_mat.shape), -999.0)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(n, col):
            for j in range(row):
                if self.chart_day_mat[i-n][j]!=0:
                    std = np.std(self.chart_day_mat[i-n:i, j])
                    vola_mat[i][j] = std/self.chart_day_mat[i][j]
        #print(vola_mat)
        vola_mat = np.delete(vola_mat, [0,1,2,3], 1) #SPY, cashなどを除く
        vola_mat = vola_mat.reshape(-1,1)
        return vola_mat

    #n日のシャープレシオ
    def ret_sharpe_ratio(self):
        sharpe_mat = np.full((self.chart_day_mat.shape), -999.0)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        return_mat = np.full((self.chart_day_mat.shape), -999.0)
        day20_ex_return = np.full((self.chart_day_mat.shape), -999.0)
        for i in range(20, col):
            for j in range(row):
                if self.chart_day_mat[i-20][j]!=0:
                    return_mat[i][j] = self.chart_day_mat[i][j]/self.chart_day_mat[i-20][j]*100-100
                    day20_ex_return[i][j] = (self.chart_day_mat[i][j]/self.chart_day_mat[i-20][j])*100-100
                    bench_mark = self.nasdaq_day_mat[i][0]/self.nasdaq_day_mat[i-20][0]*100-100 #SP500 return
                    day20_ex_return[i][j] = day20_ex_return[i][j] - bench_mark
        for i in range(220, col):
            for j in range(row):
                if self.chart_day_mat[i-220][j]!=0 and return_mat[i-200][j]!=-999:
                    risk = np.std(return_mat[i-200:i, j])
                    if risk != 0:
                        sharpe_mat[i][j] = day20_ex_return[i][j]/risk
        sharpe_mat = np.delete(sharpe_mat, [0,1,2,3], 1) #SPY, cashなどを除く
        sharpe_mat = sharpe_mat.reshape(-1,1)
        return sharpe_mat


    #過去day_range日の平均に対する、当日値の倍率
    def ret_rate_vs_mean(self, input_mat, day_range):
        volume_rate_mat = np.full(input_mat.shape, -999.0)
        for j in range(input_mat.shape[1]):
            listed = 0
            ave = 0
            for i in range(day_range, input_mat.shape[0]):
                if listed==0 and input_mat[i][j]!=0:
                    listed = 1
                if listed==1 and input_mat[i-day_range][j]!=0:
                    ave = np.mean(input_mat[i-day_range:i,j])
                    volume_rate_mat[i][j] = input_mat[i][j]/ave
        #print(volume_rate_mat)
        volume_rate_mat = volume_rate_mat.reshape(-1, 1)
        return volume_rate_mat

    def ret_RSI(self, day_range):
        chart_day_mat = np.delete(self.chart_day_mat,[0,1,2,3], 1) #SPY, cashなどを除く
        volume_day_mat = np.delete(self.volume_day_mat,[0], 1) #SPYを除く
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        rsi_mat = np.full((chart_day_mat.shape), -999)
        for i in range(day_range, col):
            for j in range(row):
                p_count = 0 #plus
                m_count = 0 #minus
                if chart_day_mat[i-day_range][j]!=0:
                    for k in range(day_range):
                        if







    #^TNXを読み込み
    def set_tnx(self, df):
        tnx_mat = df.values
        tnx_mat = tnx_mat[:,1].reshape(-1,1)
        if self.chart_day_mat.shape[0] != tnx_mat.shape[0]:
            print("chart and tnx length dont match")
            sys.exit(1)
        self.tnx_mat = tnx_mat

    #金利データ特徴量
    def ret_tnx_data(self):
        tnx_data = np.full((self.chart_day_mat.shape), -999.0)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(col):
            for j in range(row):
                tnx_data[i][j] = self.tnx_mat[i][0]
        tnx_data = np.delete(tnx_data, [0,1,2,3], 1) #SPY, cashなどを除く
        tnx_data = tnx_data.reshape(-1,1)
        return tnx_data

    #マーケットデータ(i.g.金利、SP500)のn-days変化
    def ret_market_return(self, mat, n):
        ret_data = np.full((self.chart_day_mat.shape), -999.0)
        col = self.chart_day_mat.shape[0]
        row = self.chart_day_mat.shape[1]
        for i in range(n, col):
            for j in range(row):
                ret_data[i][j] = (mat[i][0]/mat[i-n][0])*100-100
        ret_data = np.delete(ret_data, [0,1,2,3], 1)
        ret_data = ret_data.reshape(-1, 1)
        return ret_data


    def set_nasdaq_data(self, df):
        self.nasdaq_day_mat = df.values
        self.nasdaq_day_mat = np.delete(self.nasdaq_day_mat, 0, axis=1)
        self.nasdaq_day_mat = np.delete(self.nasdaq_day_mat, 0, axis=0)
        self.nasdaq_day_mat = self.nasdaq_day_mat.astype(np.float32)
        print(self.nasdaq_day_mat.shape)
        #print(self.nasdaq_day_mat)


if __name__ == "__main__":

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=2, suppress=True)

    filename = 'data/stock_daily_data.csv'
    #filename = 'data/stock_daily_data_copy.csv'
    chart_df = pd.read_csv(filename)
    Chart = Chart(chart_df)
    Chart.set_data()
    #print(Chart.chart_month_mat)
    #print(type(Chart.chart_month_mat))
    #print(type(Chart.week_index))
    #print(Chart.week_index)
    #print(Chart.month_index)
    #print(Chart.date_length)
    """
    for i in range(Chart.month_index.size):
        print(Chart.date_array[Chart.month_index[i]])
    print(type(Chart.date_array))
    """
    #print(Chart.date_array)
    #print(Chart.chart_day_mat.shape)
    #print(Chart.chart_week_mat.shape)
    #print(Chart.chart_day_mat)
    w1 = 0.6
    w2 = 0.3
    w3 = 0.3
    dPara1 = 3*20
    dPara2 = 6*20
    dPara3 = 11*20
    Chart.set_daily_rank_mat(dPara1, dPara2, dPara3, w1, w2, w3)
    print(Chart.n_day_ex_return)
    #print(Chart.daily_rank_mat)
    #print(Chart.chart_month_mat)
    #print(Chart.mom_mat.shape)
    #print(len(Chart.stock_list))
    #print(Chart.stock_list)
    #print(Chart.mom_mat)
    #print(Chart.drop_from_high_mat)
    #print(Chart.stock_list)
    #print(Chart.chart_week_mat)
    #print(Chart.mom_mat)
