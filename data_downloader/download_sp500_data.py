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
    df = x.read()
    #print(type(df))
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date']) #convert to datetime
    df['date'] = df['date'].dt.date #convert to date
    #a = df[['date', 'close']].values #convert to numpy
    a = df[['date', 'adjClose']].values #convert to numpy
    a = a.astype(np.unicode) #cast to str(unicode)
    return a


index_mat = np.array(["date","SP500","","SPY"])
index_mat = np.reshape(index_mat, (2,2))

start_date = "2003-1-1"
end_date = "2020-12-11"

sp500_mat = tiingo_download("SPY", start_date, end_date)
sp500_mat = np.concatenate([index_mat, sp500_mat], axis=0)

print(sp500_mat)
np.savetxt("../data/sp500_daily_data.csv", sp500_mat, delimiter=',', fmt='%s')


"""
try:
    例外が発生するかもしれないが、実行したい処理
except 'error name':
    例外発生時に行う処理
"""
