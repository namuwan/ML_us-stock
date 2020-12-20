import sys

import pandas as pd
import numpy as np
from scipy import signal

import ChartData as CD

class InvSim:
    """invest simlation time series"""

    #def __init__(self, mom_mat, rank_index, rank_point, bs_buy_mat, bs_sell_mat, rank_buy_th, rank_sell_th, start_month, stock_list):
    def __init__(self, CHART, MarketData, start_month, print_flag):
        #チャート(各銘柄のデータ)
        self.Chart = CHART
        self.mom_mat = CHART.mom_mat#前月比matrix
        self.rank_index = CHART.rank_index #rankIndex: chartMatをランキングしたインデックス行列 , 1列目1位，2列目2位，...
        self.daily_rank_index = CHART.daily_rank_index
        #self.rank_point = CHART.rank_point_mat
        self.daily_rank_point = CHART.daily_rank_mat
        self.bs_buy_mat = CHART.bs_buy_mat #chartMat各要素ごとに買付基準を満たしているか
        self.bs_sell_mat = CHART.bs_sell_mat #chartMat各要素ごとに売付基準に達しているか
        self.daily_bs_buy_mat = CHART.daily_bs_buy_mat #chartMat各要素ごとに買付基準を満たしているか
        self.daily_bs_sell_mat = CHART.daily_bs_sell_mat #chartMat各要素ごとに売付基準に達しているか
        self.rank_buy_th = CHART.rank_buy_thresh #rankPointの買い閾値
        self.rank_sell_th = CHART.rank_sell_thresh #rankPointの売り閾値
        self.start_month = start_month #sim開始月
        self.start_day = 20*start_month
        self.stock_list = CHART.stock_list #銘柄リスト
        self.hold_num = 5 #保有銘柄数 デフォルトは5
        self.change_num = 1 #単位時間あたりに入れ替える銘柄数 デフォルトは1
        self.col = CHART.mom_mat.shape[0] #行
        self.row = CHART.mom_mat.shape[1] #列
        self.date_length = CHART.date_length
        self.hold_stock_mat = np.full((self.date_length+1, self.hold_num), 0) #時系列ポートフォリオリスト#次月の予測含めるのでcol+1
        #その月のポートフォリオ #0行目:stockナンバー #1行目:保有量 #3行目:rank指数 #4行目：買付量 #5：損益 #6:前月比
        self.prev_day_index = 0
        self.current_day_index = 0
        self.current_day = " "
        self.result = np.ones(self.col)
        self.daily_result = np.ones(self.date_length)
        # hedge parameter
        self.hedge_on = False
        self.hedge_num = 2 #ヘッジする銘柄数
        self.hedge_count = 0
        # loss cus
        self.loss_cut_on = False
        self.lc_threshold = -30
        self.lc_new_buy_list = np.empty(0)
        #マーケット(SP500)のデータ
        self.Market = MarketData
        self.print_flag = print_flag

    initial_hold_volume = 20

    #運用simlation
    def sim_run(self, print_flag):
        self._set_start_day_stock()
        prev_month_index = 0
        for i in range(self.start_day+1, self.date_length):
            self.current_day_index = i
            self.ptfl = self.ptfl[:, np.argsort(self.ptfl[2])]
            self._daily_update_portfolio(i)
            self.daily_result[i] = sum(self.ptfl[1][:])
            self._averaging_portfolio(print_flag)
            self.ptfl = self.ptfl[:, np.argsort(self.ptfl[2])]
            if np.count_nonzero(self.Chart.month_index == i) == 1:
                if print_flag:
                    self.current_day = self.Chart.date_array[i]
                    print(self.current_day)
                    print("Total: ",end="")
                    print("{:.2f}".format(self.daily_result[i]), end="  ")
                    mom = (self.daily_result[i]/self.daily_result[prev_month_index])*100-100
                    print("mom:", end="")
                    print("{:+.2f}%".format(mom))
                    prev_month_index = i
                if print_flag:
                    self.print_this_month_holdings()
                self._change_stock(i)#銘柄入れ換え
            else:
                self.hold_stock_mat[i+1] = self.ptfl[0]
            self.prev_day_index = self.current_day_index
        print("End day: ",end="")
        print(self.Chart.date_array[self.current_day_index])
        if 0:#翌月予想
            self.print_daily_next_prediction()


    def _change_stock(self, i):
        #rankIndex行からすでにportfolioに含まれているナンバーを削除し、先頭(1位)を新たにポートフォリオ先頭に加える
        #~:否定演算子、in1d:一致するインデックスを返す
        ranklist = self.daily_rank_index[i]
        ranklist = ranklist[~np.in1d(ranklist, self.ptfl[0][self.change_num:])]
        changed_count = 0
        for j in range(self.hold_num):
            #rankpointが保有銘柄の中で最も低いか、ranksellを下回るか、bsMatsellが0かなら一旦キャッシュにする(変更フラグ)
            #if j==0 or self.ptfl[2][j]<self.rank_sell_th or self.bs_sell_mat[i][int(self.ptfl[0][j])]!=1:
            if j < self.change_num or self.ptfl[2][j]<self.rank_sell_th or self.daily_bs_sell_mat[i][int(self.ptfl[0][j])]!=1:
                not_changed = False
                while changed_count < self.row-self.hold_num:
                    if self.daily_rank_point[i][ranklist[changed_count]]>self.rank_buy_th and self.daily_bs_buy_mat[i][ranklist[changed_count]]==1:
                        self.hold_stock_mat[i+1][j] = ranklist[changed_count]
                        if self.print_flag==1:
                            print("{:>7}".format(self.stock_list[int(self.ptfl[0][j])][:7]), end=' ')#7文字目まで、右詰め>
                            print(" ==> ", end=' ')
                            print("{:>7}".format(self.stock_list[int(ranklist[changed_count])][:7]) )#7文字目まで、右詰め>
                        changed_count += 1
                        not_changed = True
                        break
                    changed_count += 1
                if not_changed==False:
                    if self.ptfl[2][j]>=1.5:
                        self.hold_stock_mat[i+1][j] = self.ptfl[0][j]
                    else:
                        self.hold_stock_mat[i+1][j] = 0 #[0]はキャッシュ
            else:
                self.hold_stock_mat[i+1][j] = self.ptfl[0][j]
        if self.hedge_on == True:
            if self.Market.hedge_indicator[i]==True:#ヘッジ期間の場合は0,1はキャッシュ
                for l in range(self.hedge_num):
                    self.hold_stock_mat[i+1][l] = 2 #no.2はインバース
        if self.print_flag==1:
            print()


    #ptflを更新
    def _daily_update_portfolio(self, i):
        for j in range(self.hold_num):#毎日の更新
            if self.ptfl[0][j]!=0: #cash(0)でない場合
                self.ptfl[1][j] *= self.Chart.chart_day_mat[i][int(self.ptfl[0][j])] / self.Chart.chart_day_mat[i-1][int(self.ptfl[0][j])]
                self.ptfl[2][j] = self.daily_rank_point[i][int(self.ptfl[0][j])]
                self.ptfl[4][j] = self.ptfl[1][j]/self.ptfl[3][j]*100-100 #損益
                self.ptfl[5][j] = self.Chart.chart_day_mat[i][int(self.ptfl[0][j])] / self.Chart.chart_day_mat[i-20][int(self.ptfl[0][j])] *100-100
                if self.ptfl[6][j] < self.ptfl[1][j]:
                    self.ptfl[6][j] = self.ptfl[1][j] #高値更新
                self.ptfl[7][j] = self.ptfl[1][j]/self.ptfl[6][j]*100-100 #保有中高値からの下落率
                if self.ptfl[8][j] < self.ptfl[2][j]:
                    self.ptfl[8][j] = self.ptfl[2][j]
                self.ptfl[9][j] = self.ptfl[2][j]/self.ptfl[8][j]*100-100 #保有中高値からの下落率
            if self.loss_cut_on == True:
                self._loss_cut_operation(i, j)
        if self.hedge_on == True:
            self._hedge_operation(i)
        #month_indexの中にiと一致するものがあれば新規購入銘柄をセット
        if np.count_nonzero(self.Chart.month_index == i-1) == 1:#前日が締日の日であれば、ポートフォリオ変更
            for j in range(self.hold_num):
                self.ptfl[0][j] = self.hold_stock_mat[i][j] #銘柄更新
                if np.count_nonzero(self.hold_stock_mat[i-1]==int(self.ptfl[0][j]))==0:
                    self.ptfl[3][j] = self.ptfl[1][j] #買付量をセット
                    self.ptfl[4][j] = 0 #損益をリセット
                    self.ptfl[6][j] = 0 #高値をリセット
                    self.ptfl[8][j] = 0 #rank高値をリセット
                if self.ptfl[0][j] == 0:
                    self.ptfl[2][j] = 1.2
                    self.ptfl[5][j] = 0
                    self.ptfl[6][j] = 0 #保有中高値
                    self.ptfl[7][j] = 0 #保有中高値からの下落率
                    self.ptfl[8][j] = 1.2
                    self.ptfl[9][j] = 0


    def _hedge_operation(self, i):
        if self.Market.hedge_indicator[i-1]==True and self.Market.hedge_indicator[i-2]==False:#「前日に」ヘッジインジケータが0から1に変化
            self.hedge_count = self.hedge_count + 1 #全期間で何回ヘッジが発動したか
            if self.print_flag:
                print("Break below the hedge line. Sell two stocks.",end=" ")
                print(self.Chart.date_array[i])
            self.ptfl = self.ptfl[:, np.argsort(self.ptfl[2])]
            for l in range(self.hedge_num):
                self.ptfl[0][l] = 2 #no.2は-1倍インバースを意味する
                self.ptfl[2][l] = 1.2
                self.ptfl[3][l] = self.ptfl[1][l]
                self.ptfl[5][l] = 0
                self.ptfl[6][l] = self.ptfl[1][l] #保有中高値
                self.ptfl[7][l] = 0 #保有中高値からの下落率
                self.ptfl[8][l] = self.ptfl[2][l] #保有中rank高値
                self.ptfl[9][l] = 0 #保有中rank高値からの下落率
        if self.Market.hedge_indicator[i-1]==False and self.Market.hedge_indicator[i-2]==True:#「前日に」ヘッジインジケータが1から0に変化
            if self.print_flag:
                print("Break up the hedge line. Buy two stocks.",end=" ")
                print(self.Chart.date_array[i])
            self.ptfl = self.ptfl[:, np.argsort(self.ptfl[0])] #銘柄番号基準でソート。キャッシュが手前にくる
            ranklist = self.daily_rank_index[i-1][1:]#1位は回避
            ranklist = ranklist[~np.in1d(ranklist, self.ptfl[0][self.change_num:])]#既保有銘柄は除外
            changed_count = 0
            new_buy_list = [0, 0, 0]
            inverse_count = 0
            for l in range(self.hedge_num):
                while changed_count < self.row-self.hold_num:
                    if self.daily_rank_point[i][ranklist[changed_count]]>self.rank_buy_th and self.daily_bs_buy_mat[i][ranklist[changed_count]]==1:
                        new_buy_list[l] = ranklist[changed_count]
                        changed_count += 1
                        break
                    changed_count += 1
                while inverse_count < self.hold_num:
                    if self.ptfl[0][inverse_count] == 2: #no.2はインバース
                        self.ptfl[0][inverse_count] = new_buy_list[l]
                        self.ptfl[3][inverse_count] = self.ptfl[1][inverse_count] #買付量をセット
                        self.ptfl[4][inverse_count] = 0 #損益をリセット
                        self.ptfl[6][inverse_count] = 0 #高値をリセット
                        self.ptfl[8][inverse_count] = 0
                        inverse_count += 1
                        break
                    inverse_count += 1
            self.ptfl = self.ptfl[:, np.argsort(self.ptfl[2])] #rank基準でソートし直し

    def _loss_cut_operation(self, i, j):
        rank_list = self.daily_rank_index[i]
        rank_list = rank_list[~np.in1d(rank_list, self.ptfl[0])]
        rank_list = rank_list[~np.in1d(rank_list, self.lc_new_buy_list)]
        top_rank = rank_list[0]
        top_rank_score = self.daily_rank_point[i][top_rank]
        if self.ptfl[10][j] == 1:#翌終値でロスカット行使
            print(self.Chart.date_array[i], end=' ')
            print("    loss cut !", end=' ')
            print("{:>7}".format(self.stock_list[int(self.ptfl[0][j])][:7]), end=' ')#7文字目まで、右詰め>
            print( "{:>3.02f}".format(self.daily_rank_point[i][int(self.ptfl[0][j])]), end=' ')
            print(" new Buy ", end=' ')
            print("{:>7}".format(self.stock_list[int(self.lc_new_buy_list[0])][:7]), end=' ')#7文字目まで、右詰め>
            print( "{:>3.02f}".format(self.daily_rank_point[i][int(self.lc_new_buy_list[0])]))
            #ロスカット作業
            #self.ptfl[0][j] = 0; self.ptfl[2][j] = 1.2
            self.ptfl[0][j] = self.lc_new_buy_list[0]
            self.ptfl[2][j] = self.daily_rank_point[i][int(self.lc_new_buy_list[0])]
            self.ptfl[3][j] = self.ptfl[1][j] ;self.ptfl[5][j] = 0
            self.ptfl[6][j] = self.ptfl[1][j] #保有中高値
            self.ptfl[7][j] = 0 #保有中高値からの下落率
            self.ptfl[8][j] = self.ptfl[2][j] #保有中高値
            self.ptfl[9][j] = 0 #保有中高値からの下落率
            self.ptfl[10][j] = 0
            self.lc_new_buy_list = np.delete(self.lc_new_buy_list, 0) #先頭を削除
        #if self.ptfl[7][j] <= self.lc_threshold:#保有中高値から**%下落したらロスカット発動
        if self.ptfl[2][j] <= top_rank_score-0.5 and self.daily_rank_point[i][top_rank]>self.rank_buy_th and self.daily_bs_buy_mat[i][top_rank]==1:
        #if self.ptfl[9][j] <= self.lc_threshold:#保有中高値から**%下落したらロスカット発動
            #print("{:>7}".format(self.stock_list[int(top_rank)][:7]), end=' ')#7文字目まで、右詰め>
            #print(top_rank_score)
            self.ptfl[10][j] = 1
            """
            print(self.Chart.date_array[i], end=' ')
            print("pre loss cut !", end=' ')
            print("{:>7}".format(self.stock_list[int(self.ptfl[0][j])][:7]), end=' ')#7文字目まで、右詰め>
            print( "{:>3.02f}".format(self.daily_rank_point[i][int(self.ptfl[0][j])]), end=' ')
            print(" new Buy ", end=' ')
            print("{:>7}".format(self.stock_list[int(top_rank)][:7]), end=' ')#7文字目まで、右詰め>
            print( "{:>3.02f}".format(self.daily_rank_point[i][top_rank]))
            """
            self.lc_new_buy_list = np.insert(self.lc_new_buy_list, 0, int(top_rank))


    #開始日のポートフォリオを設定,単純にランク順(ただしbuymatの閾値条件はあり)
    def _set_start_day_stock(self):
        count, j = 0, 0
        while j < self.row:
            if self.daily_bs_buy_mat[self.start_day-1][self.daily_rank_index[self.start_day-1][j]] > 0:
                self.hold_stock_mat[self.start_day][count] = self.daily_rank_index[self.start_day-1][j]
            count += 1
            if count == self.hold_num:
                break
            j += 1
        for i in range(self.hold_num):
            self.ptfl[0][i] = self.hold_stock_mat[self.start_day][i] #銘柄番号
            self.ptfl[1][i] = self.initial_hold_volume #保有額
            self.ptfl[2][i] = self.daily_rank_point[self.start_day][int(self.ptfl[0][i])] #ランクポイント
            self.ptfl[3][i] = self.initial_hold_volume #買付量
            self.ptfl[4][i] = 0
        self.ptfl = self.ptfl[:, np.argsort(self.ptfl[2])] #rankPointの最下位を先頭にソートする
        #temp = np.zeros((2, self.hold_num)) #2行追加
        #self.ptfl = np.concatenate([self.ptfl, temp])
        for k in range(self.start_day+1):
            self.daily_result[k] = 100


    def _inv_one_stock_for_month(self, prev_day, current_day, value, cost, stock_no, high_price):
        lc_line_over = 0 #loss cut
        drop_line_over = 0
        drop_line = -10000
        for i in range(prev_day, current_day):
            value *= self.Chart.chart_day_mat[i+1][stock_no] / self.Chart.chart_day_mat[i][stock_no]
            pl = value/cost*100-100
            #"""
            #高値からの下落率が一定値を超えたら損切り
            if drop_line_over == 1: #損切りラインを超えたら、翌日終値でロスカット
                print(self.stock_list[stock_no], end=" ")
                print("  loss cut")
                break
            if high_price < value:
                high_price = value #高値を更新
            #"""
            drop = value/high_price*100-100
            if drop < drop_line:
                drop_line_over = 1
            if pl < -10:
                lc_line_over = 1
                #print(self.stock_list[stock_no], end=" ")
        return(value, high_price)

    #ptflの保有額を平均化
    def _averaging_portfolio(self, print_flag):
        self.ptfl = self.ptfl[:, np.argsort(self.ptfl[1])] #ptflを保有額でソートする
        if self.ptfl[1][-1]/self.ptfl[1][0] > 2: #保有額の最小値と最大値の比が2を超えている場合、是正
            temp = self.ptfl[1][0] + self.ptfl[1][-1] #保有額の最大値を最小値の和
            self.ptfl[3][0] = temp/2 * ( 100 / (100+self.ptfl[4][0]) )
            self.ptfl[3][-1] = temp/2 * ( 100 / (100+self.ptfl[4][-1]) )
            self.ptfl[6][0] = self.ptfl[6][0] * (temp/2)/self.ptfl[1][0]
            self.ptfl[6][-1] = self.ptfl[6][-1] * (temp/2)/self.ptfl[1][-1]
            self.ptfl[1][0] = temp/2
            self.ptfl[1][-1] = temp/2
            if print_flag:
                print("**Averaging**", end=' ')
                print(self.stock_list[int(self.ptfl[0][0])], end=' ')#7文字目まで
                print("& ", end=' ')
                print(self.stock_list[int(self.ptfl[0][-1])]) #7文字目まで
                print()

    #当月のポートフォリオを表示
    def print_this_month_holdings(self):
        print("      ", end=' ')
        for i in range(self.hold_num):
            if self.ptfl[0][i]!=0 and self.ptfl[0][i]!=-2:
                print("{:>7}".format(self.stock_list[int(self.ptfl[0][i])][:7]), end=' ')#7文字目まで、右詰め>
            elif self.ptfl[0][i]==0:
                print("{:>7}".format("cash"), end=' ')#7文字目まで
            elif self.ptfl[0][i]==-2:
                print("{:>7}".format("d_bear"), end=' ')#7文字目まで
        print()
        #print(self.ptfl[1:][:]) #保有額とrankPointを表示
        print("{:>6}".format("value:"), end=" ")
        for j in range(self.hold_num):
            print( "{:>7.02f}".format((self.ptfl[1][j])), end=' ')
        print(" ")
        print("{:>6}".format("rank:"), end=" ")
        for j in range(self.hold_num):
            print( "{:>7.02f}".format((self.ptfl[2][j])), end=' ')
        print(" ")
        print("{:>6}".format("cost:"), end=" ")
        for j in range(self.hold_num):
            print( "{:>7.02f}".format((self.ptfl[3][j])), end=' ')
        print(" ")
        print("{:>6}".format("P/L:"), end=" ")
        for j in range(self.hold_num):
            print( "{:>6.02f}%".format((self.ptfl[4][j])), end=' ')
        print(" ")
        print("{:>6}".format("mom:"), end=" ")
        for j in range(self.hold_num):
            print( "{:>6.02f}%".format((self.ptfl[5][j])), end=' ')
        print(" ")
        print("{:>6}".format("high:"), end=" ")
        for j in range(self.hold_num):
            print( "{:>7.02f}".format((self.ptfl[6][j])), end=' ')
        print(" ")
        print("{:>6}".format("drop:"), end=" ")
        for j in range(self.hold_num):
            print( "{:>6.02f}%".format((self.ptfl[7][j])), end=' ')
        print()
        print("{:>6}".format("r-hgh:"), end=" ")
        for j in range(self.hold_num):
            print( "{:>7.02f}".format((self.ptfl[8][j])), end=' ')
        print(" ")
        print("{:>6}".format("r-drp:"), end=" ")
        for j in range(self.hold_num):
            print( "{:>6.02f}%".format((self.ptfl[9][j])), end=' ')
        print()

    #翌月の予想を表示
    def print_next_prediction(self):
        print()
        print("*** NextMonth ***  ",end="")
        for i in range(self.hold_num):
            if self.hold_stock_mat[self.col][i] != 0:
                print("{:<8}".format(self.stock_list[int(self.hold_stock_mat[self.col][i])][:8]), end='') #7文字目まで表示
                print("[" + str(self.hold_stock_mat[self.col][i]) + "]", end=' ')
            else:
                print("{:<8}".format("cash"), end=' ')#7文字目まで
                print()
        print()
        print()

    def print_daily_next_prediction(self):
        print()
        print("*** NextMonth ***  ",end="")
        for i in range(self.hold_num):
            if self.hold_stock_mat[self.date_length][i] != 0:
                #print("{:<8}".format(self.stock_list[int(self.hold_stock_mat[self.date_length][i])][:8]), end='') #7文字目まで表示
                print("{:<8}".format(self.stock_list[int(self.hold_stock_mat[-1][i])][:8]), end='') #7文字目まで表示
                #print("[" + str(self.hold_stock_mat[self.date_length][i]) + "]", end=' ')
                print("[" + str(self.hold_stock_mat[-1][i]) + "]", end=' ')
            else:
                print("{:<8}".format("cash"), end=' ')#7文字目まで
                print()
        print()
        print()


    def print_result(self):
        print("### OPERATION RESULT ###")
        print(self.result)
        print()

    def print_daily_result(self):
        print("### OPERATION RESULT ###")
        """
        for i in range(self.Chart.date_length):
            if np.count_nonzero(self.Chart.month_index == i) == 1:
                print(self.daily_result[i])
        """
        list_month_index = self.Chart.month_index.tolist()
        print(self.daily_result[list_month_index])
        print()


    def set_hold_num(self, a):
        self.hold_num = a
        #self.hold_stock_mat = np.full((self.col+1, self.hold_num), -1)
        self.hold_stock_mat = np.full((self.date_length+1, self.hold_num), 0)
        #0:name, 1:value, 2:rank, 3:cost, 4:P/L, 5:mom, 6:high, 7:drop, 8:rank-high, 9:rank-drop, 10:losscut-flag
        self.ptfl = np.zeros((11, self.hold_num))
        self.initial_hold_volume = 100/a
        self.lc_control = np.full(self.hold_num, 0)

    def set_change_num(self, a):
        self.change_num = a


### old codes ###
    """
    #月更新のsim_run
    def sim_run(self, print_flag):
        self._set_start_momth_stock()
        for i in range(self.start_month+1, self.col):
            self.current_day_index = self.Chart.month_index[i]
            self._update_portfolio(i) #ptflを更新
            self.result[i] = sum(self.ptfl[1][:]) #保有額総額を入力
            self._averaging_portfolio(print_flag) #保有額の差が大きくなると是正、ptflの順番も変わる
            self.ptfl = self.ptfl[:, np.argsort(self.ptfl[2])] #ptflをrankPoint行基準で昇順にソートする(先頭はrankpoint小さい)
            if print_flag:
                self.current_day = self.Chart.date_array[self.current_day_index]
                print(self.current_day)
                print("Total: ",end="")
                print("{:.2f}".format(self.result[i]), end="  ")
                mom = (self.result[i]/self.result[i-1])*100-100
                print("mom:", end="")
                print("{:+.2f}%".format(mom))
            if print_flag:#ポートフォリオを表示
                self.print_this_month_holdings()
            #銘柄入れ替え
            self.operation(i)
            self.prev_day_index = self.current_day_index
    """

    """
    #運用algorithm
    def operation(self, i):
        #rankIndex行からすでにportfolioに含まれているナンバーを削除し、先頭(1位)を新たにポートフォリオ先頭に加える
        #~:否定演算子、in1d:一致するインデックスを返す #[1:]のところi.e.先頭値も候補
        ranklist = self.rank_index[i][1:]#1位は回避
        ranklist = ranklist[~np.in1d(ranklist, self.ptfl[0][self.change_num:])]
        changed_count = 0
        for j in range(self.hold_num):
            #rankpointが保有銘柄の中で最も低いか、ranksellを下回るか、bsMatsellが0かなら一旦キャッシュにする(変更フラグ)
            #if j==0 or self.ptfl[2][j]<self.rank_sell_th or self.bs_sell_mat[i][int(self.ptfl[0][j])]!=1:
            if j < self.change_num or self.ptfl[2][j]<self.rank_sell_th or self.bs_sell_mat[i][int(self.ptfl[0][j])]!=1:
                not_changed = False
                while changed_count < self.row-self.hold_num:
                    if self.rank_point[i][ranklist[changed_count]]>self.rank_buy_th and self.bs_buy_mat[i][ranklist[changed_count]]==1:
                        self.hold_stock_mat[i+1][j] = ranklist[changed_count]
                        changed_count += 1
                        not_changed = True
                        break
                    changed_count += 1
                if not_changed==False:
                    if self.ptfl[2][j]>=1.5:
                        self.hold_stock_mat[i+1][j] = self.ptfl[0][j]
                    else:
                        self.hold_stock_mat[i+1][j] = -1 #-1はキャッシュ
            else:
                self.hold_stock_mat[i+1][j] = self.ptfl[0][j]


    def _set_start_momth_stock(self):
        count, j = 0, 0
        while j < self.row:
            if self.bs_buy_mat[self.start_month-1][self.rank_index[self.start_month-1][j]] > 0:
                self.hold_stock_mat[self.start_month][count] = self.rank_index[self.start_month-1][j]
            count += 1
            if count == self.hold_num:
                break
            j += 1
        for i in range(self.hold_num):
            self.ptfl[0][i] = self.hold_stock_mat[self.start_month][i] #銘柄番号
            self.ptfl[1][i] = self.initial_hold_volume #保有額
            self.ptfl[2][i] = self.rank_point[self.start_month][int(self.ptfl[0][i])] #ランクポイント
            self.ptfl[3][i] = self.initial_hold_volume #買付量
            self.ptfl[4][i] = 0
        self.ptfl = self.ptfl[:, np.argsort(self.ptfl[2])] #rankPointの最下位を先頭にソートする
        temp = np.zeros((2, self.hold_num)) #2行追加
        self.ptfl = np.concatenate([self.ptfl, temp])
        for k in range(self.start_month+1):
            self.result[k] = 100



    #ptflを更新
    def _update_portfolio(self, i):
        #新規購入銘柄をセット
        for j in range(self.hold_num):
            self.ptfl[0][j] = self.hold_stock_mat[i][j] #銘柄更新
            if np.count_nonzero(self.hold_stock_mat[i-1]==int(self.ptfl[0][j]))==0: #self.ptfl[0][j]が1個前のhold_stock_matになかったら
                self.ptfl[3][j] = self.ptfl[1][j] #買付量をセット
                self.ptfl[4][j] = 0 #損益をリセット
                self.ptfl[6][j] = 0 #高値をリセット
            if self.ptfl[0][j] == -1:
                self.ptfl[2][j] = 1.2
                self.ptfl[5][j] = 0
                self.ptfl[6][j] = 0 #保有中高値
                self.ptfl[7][j] = 0 #保有中高値からの下落率
        for k in range(self.prev_day_index, self.current_day_index):
            for j in range(self.hold_num):
                if self.ptfl[0][j]!=-1 and self.ptfl[0][j]!=-2: #cash(-1)、double_bear(-2)でない場合
                    self.ptfl[1][j] *= self.Chart.chart_day_mat[k+1][int(self.ptfl[0][j])] / self.Chart.chart_day_mat[k][int(self.ptfl[0][j])]
                    #TODO デイリーランク更新
                    self.ptfl[4][j] = self.ptfl[1][j]/self.ptfl[3][j]*100-100 #損益
                    if self.ptfl[6][j] < self.ptfl[1][j]:
                        self.ptfl[6][j] = self.ptfl[1][j] #高値更新
                    self.ptfl[7][j] = self.ptfl[1][j]/self.ptfl[6][j]*100-100 #保有中高値からの下落率
                elif self.ptfl[0][j] == -2:
                    self.ptfl[1][j] *= self.Market.double_bear[k+1]/self.Market.double_bear[k]
                    self.ptfl[4][j] = self.ptfl[1][j]/self.ptfl[3][j]*100-100
                    if self.ptfl[6][j] < self.ptfl[1][j]:
                        self.ptfl[6][j] = self.ptfl[1][j]
                    self.ptfl[7][j] = self.ptfl[1][j]/self.ptfl[6][j]*100-100
            #if self.Market.hedge_indicator[k] == 0:
            #    for l in range(2):
            #        self.ptfl[0][l] = -2 #"-2"はダブルベア
            #        self.ptfl[2][l] = 1.2
            #        self.ptfl[3][l] = self.ptfl[1][l]
            #        self.ptfl[5][l] = 0
            #        self.ptfl[6][l] = 0 #保有中高値
            #        self.ptfl[7][l] = 0 #保有中高値からの下落率
        for j in range(self.hold_num):
            if self.ptfl[0][j]!=-1 and self.ptfl[0][j]!=-2: #cash(-1)でない場合
                self.ptfl[2][j] = self.rank_point[i][int(self.ptfl[0][j])] #rankPointを更新
                self.ptfl[5][j] = self.mom_mat[i][int(self.ptfl[0][j])]*100-100 #前月比

"""
