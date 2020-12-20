import pandas as pd
import numpy as np
from scipy import signal

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as dates

import my_function_individual as mfiv

class Market:
    """Market Data"""
    def __init__(self, df_sp500):
        #self.date_mat = chart_df.iloc[:,0:1].values
        self.chart = df_sp500.values
        self.chart = np.delete(self.chart, 0, axis=1)
        self.chart = np.delete(self.chart, 0, axis=0)
        self.chart = self.chart.astype(np.float32)
        self.length = self.chart.size
        ## TODO:
        #迅速な損切りを目標としているので月足ではだめ
        #日足でうごかないとだめ
        self.ma_12m_mat = np.zeros(self.chart.size)
        self.ma_15d_mat = np.zeros(self.chart.size)
        self.ma_5d_mat = np.zeros(self.chart.size)
        #self.ma12m_bs = np.zeros(self.chart.size)
        #self.ma24_bs = np.zeros(chart_SP500.size)
        self.double_bear = np.zeros(self.chart.size)
        self.triple_bear = np.zeros(self.chart.size)

    def set(self):
        #default: 150,25,5
        self.ma_12m_mat = mfiv.returnSMA(self.chart, 200)
        self.ma_15d_mat = mfiv.returnSMA(self.chart, 50)
        self.ma_5d_mat = mfiv.returnSMA(self.chart, 10)
        #self.ma_24w_mat = mfiv.returnSMA(self.chart, 24)
        self.ma12m_bs = self._comp_sma(self.chart, self.ma_12m_mat)
        self.ma5_vs_15 = self._comp_sma(self.ma_5d_mat, self.ma_15d_mat)

        #0なら通常、1ならヘッジ
        self.hedge_indicator = np.logical_not(np.logical_or(self.ma12m_bs ,self.ma5_vs_15))

        self._set_bear()

    #aがbより上だったら1、下だったら-1
    def _comp_sma(self, a, b):
        bs_mat = np.zeros(a.shape)
        row = a.shape[0]
        for j in range(row):
            if b[j] != 0:
                comp = a[j] / b[j]
                if comp > 1:
                    bs_mat[j] = 1
                else:
                    bs_mat[j] = 0
        return(bs_mat)

    def _set_bear(self):
        self.double_bear[0] = 10000
        self.triple_bear[0] = 10000
        for i in range(1, self.chart.shape[0]):
            self.double_bear[i] = self.double_bear[i-1] * (1-2*(1-(self.chart[i]/self.chart[i-1])))
            self.triple_bear[i] = self.triple_bear[i-1] * (1-3*(1-(self.chart[i]/self.chart[i-1])))


if __name__ == "__main__":
    filename = 'data/sp500_daily_data.csv' #日次
    sp500_df = pd.read_csv(filename)
    M = Market(sp500_df)
    M.set()
    #print(M.ma12m_bs.size)
    #print(M.ma12m_bs)
    np.set_printoptions(threshold=np.inf)
    #mfiv.show_plot(M.chart)
    #mfiv.show_plots(M.chart, M.ma_12m_mat, M.ma_15d_mat, M.ma_5d_mat)

    plt.yscale("log")
    plt.plot(M.chart[3800:])
    plt.plot(M.ma_12m_mat[3800:])
    #"""
    for i in range(3800, M.hedge_indicator.size):
        if M.hedge_indicator[i]==0:
            plt.axvspan(i-3800, i+1-3800, color="yellow")
    #"""
    print(M.hedge_indicator)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
