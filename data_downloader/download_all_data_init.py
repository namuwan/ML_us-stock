from pandas_datareader.tiingo import TiingoDailyReader
import pandas as pd
import numpy as np
import csv

#import my_function_individual as mfiv

from datetime import datetime as dt


def tiingo_download(ticker, start_date, end_date):
    x = TiingoDailyReader(ticker,
                            start_date,
                            end_date,
                            api_key="4c67c00d97ae38d28de43dc90e0c3ef25471cdef")
    #df = x.read_csv
    df = x.read()
    #print(type(df))
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date']) #convert to datetime
    df['date'] = df['date'].dt.date #convert to date
    #a = df[['date', 'close']].values #convert to numpy
    a = df[['date', 'adjClose']].values #convert to numpy
    a = a.astype(np.unicode) #cast to str(unicode)
    return a


filename = '../data/ticker_list.csv'
start_date = "2003-1-1"
end_date = "2020-6-27"
#ticker_list_df = pd.read_csv(filename)
with open(filename) as f:
    reader = csv.reader(f)
    ticker_list = [row for row in reader]

#print(ticker_list)
#print(len(ticker_list[0]))
ticker_mat = np.array(ticker_list)
temp = np.array(["date",""])
temp = np.reshape(temp, (2,1))
index_mat = np.concatenate([temp, ticker_mat], axis=1)
#print(ticker_mat)
#print(ticker_mat.shape)

all_mat = tiingo_download("SPY", start_date, end_date)
all_mat = np.delete(all_mat, 1, 1)
data_len = len(all_mat)

for i in range(len(ticker_list[0])):
    print(ticker_list[0][i])
    a = tiingo_download(ticker_list[0][i], start_date, end_date)
    a = np.delete(a, 0, 1)
    if a.size < data_len:
        size_less_num = data_len - a.size
        fill_zero_mat = np.zeros(size_less_num)
        fill_zero_mat = np.reshape(fill_zero_mat, (fill_zero_mat.size, 1))
        a = np.concatenate([fill_zero_mat, a], axis=0)
    all_mat = np.concatenate([all_mat, a], axis=1)
print(all_mat.shape)
print(index_mat.shape)
all_mat = np.concatenate([index_mat, all_mat], axis=0)
#all_mat = np.concatenate([index_mat[:,0:6], all_mat], axis=0)
print(all_mat)

np.savetxt("../data/stock_daily_data.csv", all_mat, delimiter=',', fmt='%s')
