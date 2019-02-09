#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
 
Learningスクリプト用関数 
 
"""


import os
import numpy as np
import time
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.externals import joblib


def readData(input_path):
    print("\n・データ読み込み")
    time.sleep(0.5)
    
    milage_list = list(np.load(f"{input_path}/milage_list.npz")["arr_0"])
    
    train_data_dict = {}
    n_orgdata_dict = {}
    
    for milage in tqdm(milage_list):
        # 訓練データを入力
        train_data_dict[milage] = (np.load(f"{input_path}/{milage}/train_data.npz")["arr_0"], # y
                                   np.load(f"{input_path}/{milage}/train_data.npz")["arr_1"]) # X
                
        # 直近180日間の原系列のデータ数を保存
        n_orgdata_dict[milage] = int(np.load(f"{input_path}/{milage}/n_orgdata.npz")["arr_0"])
        
    return train_data_dict, n_orgdata_dict


def train(train_data_dict, ridge_alpha, n_orgdata_dict, output_path):
    print("\n・学習モデル作成")
    time.sleep(0.5)
    
    for milage in tqdm(list(train_data_dict.keys())):
        # アウトプットフォルダを作成
        tmp_output_path = f"{output_path}/{milage}"
        if os.path.exists(tmp_output_path)==False: os.mkdir(tmp_output_path)
        
        # 訓練データを取得
        y, X = train_data_dict[milage]
        
        # モデル学習・逐次予測(訓練データ数が10以上，直近180日間の原系列数が10以上のものに限りモデル作成)
        if X.shape[0] >= 10 and n_orgdata_dict[milage] >= 10:
            # モデル初期化・学習
            model = Ridge(alpha=ridge_alpha)
            model.fit(X=X, y=y)
            
            # モデル保存
            joblib.dump(model, f"{tmp_output_path}/ARIMA.pkl")