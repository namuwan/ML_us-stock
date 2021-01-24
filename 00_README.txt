make_train_data
：train用 result+特徴量csv を出力する。

train
：訓練

test
；テスト

main_test
：modelとデータを読み込んで全データのpredictをする

feature_selection.py
：make_train_data.pyで生成した特徴量.csvから、
　正解との相関係数の高いものだけに間引く。
