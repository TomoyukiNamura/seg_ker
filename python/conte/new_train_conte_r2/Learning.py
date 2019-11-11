#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Learning

"""


import os
os.chdir("/Users/tomoyuki/python_workspace/new_train_conte")

import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import time
import configparser
from sklearn.linear_model import Ridge
from sklearn.externals import joblib

from func import model_funcs
from func import post_funcs



for track in ["A","B","C","D"]:
    
    print(f"\ntrack{track} ===============================")
    time.sleep(0.5)
    
    ## 初期処理 
    ## 設定ファイル読み込み
    config = configparser.ConfigParser()
    config.read(f"conf/track_{track}.ini", 'UTF-8')
    
    # [model]
    
    
    ## 予測対象キロ程，予測期間の設定
    t_pred = 91  # 変更しない
    
    # (追加)データ読み込み
    input_path = f"output/Preprocessing/track_{track}"
    milage_list = list(np.load(f"{input_path}/milage_list.npz")["arr_0"])
    
    train_data_dict = {}
    n_orgdata_dict = {}
    
    for milage in tqdm(milage_list):
        # 訓練データを入力
        train_data_dict[milage] = (np.load(f"{input_path}/{milage}/train_data.npz")["arr_0"], # y
                                   np.load(f"{input_path}/{milage}/train_data.npz")["arr_1"]) # X
                
        # 直近180日間の原系列のデータ数を保存
        n_orgdata_dict[milage] = int(np.load(f"{input_path}/{milage}/n_orgdata.npz")["arr_0"])
        
    
    ## 学習
    print("\n・学習モデル作成 ===============================")
    time.sleep(0.5)
    
    
    for milage in tqdm(list(train_data_dict.keys())):
        # アウトプットフォルダを作成
        output_path = f"output/Learning/track_{track}/{milage}"
        if os.path.exists(output_path)==False: os.mkdir(output_path)
        
        # 訓練データを取得
        y, X = train_data_dict[milage]
        
        # モデル学習・逐次予測(訓練データ数が10以上，直近180日間の原系列数が10以上のものに限りモデル作成)
        if X.shape[0] >= 10 and n_orgdata_dict[milage] >= 10:
            # モデル初期化・学習
            model = Ridge(alpha=0.5)
            model.fit(X=X, y=y)
            
            # モデル保存
            joblib.dump(model, f"{output_path}/ARIMA.pkl")

            
            
    