#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Learning

"""


import os
os.chdir("/Users/tomoyuki/python_workspace/new_train_conte")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import time
import shutil
import configparser

from func import prior_funcs
from func import model_funcs
from func import post_funcs

        
#        aaa = np.load(f"{output_path}/init_raw.npz")["arr_0"]
#        bbb = np.load(f"{output_path}/init_diff.npz")["arr_0"]
#        ccc = np.load(f"{output_path}/train_data.npz")["arr_0"]
#        ddd = np.load(f"{output_path}/train_data.npz")["arr_1"]


for track in ["A","B","C","D"]:
    
    print(f"\ntrack{track} ===============================")
    time.sleep(0.5)
    
        ## 初期処理 
    ## 設定ファイル読み込み
    config = configparser.ConfigParser()
    config.read(f"conf/track_{track}.ini", 'UTF-8')
    
    # [prior]
    tol_sigma_raw_prior = config.getfloat('prior', 'tol_sigma_raw_prior')
    window = config.getint('prior', 'window')
    min_periods = config.getint('prior', 'min_periods')
    center = config.getboolean('prior', 'center')
    tol_sigma_diff_prior = config.getfloat('prior', 'tol_sigma_diff_prior')
    window_diff = config.getint('prior', 'window_diff')
    min_periods_diff = config.getint('prior', 'min_periods_diff')
    center_diff = config.getboolean('prior', 'center_diff')
    start_period = config.getint('prior', 'start_period')
    n_average_date = config.getint('prior', 'n_average_date')
    start_average_method = config.get('prior', 'start_average_method')
    
    # [model]
    train_date_id_list = list(range(config.getint('model', 'train_date_id_start'), config.getint('model', 'train_date_id_end')))
    model_name_pred = config.get('model', 'model_name_pred')
    n_diff = config.getint('model', 'n_diff')
    
    # [post]
    tol_abnormal_max_min = config.getfloat('post', 'tol_abnormal_max_min')
    tol_abnormal_upper = config.getfloat('post', 'tol_abnormal_upper')
    tol_abnormal_lower = config.getfloat('post', 'tol_abnormal_lower')
    method_post = config.get('post', 'method_post')    
    
    
    # 高低左データ(行：時間，列：空間)読み込み
    df_irregularity = pd.read_csv(f"input/irregularity_{track}.csv")
    
    
    ## 予測対象キロ程，予測期間の設定
    start_date_id=df_irregularity.shape[0] -1 # start_date_id日目の原系列，差分系列を初期値とする＝＞start_date_id+1日目から予測
    t_pred = 91  # 変更しない
    
    
    # (追加)データ読み込み
    input_path = f"output/Preprocessing/track_{track}"
    milage_list = list(np.load(f"{input_path}/milage_list.npz")["arr_0"])
    
    init_raw_dict = {}
    init_diff_dict = {}
    train_data_dict = {}
    n_orgdata_dict = {}
    
    for milage in tqdm(milage_list):
        # 初期原データを入力
        init_raw_dict[milage] = float(np.load(f"{input_path}/{milage}/init_raw.npz")["arr_0"])
        
        # 初期差分を入力
        init_diff_dict[milage] = np.load(f"{input_path}/{milage}/init_diff.npz")["arr_0"]
        
        # 訓練データを入力
        train_data_dict[milage] = (np.load(f"{input_path}/{milage}/train_data.npz")["arr_0"], # y
                                   np.load(f"{input_path}/{milage}/train_data.npz")["arr_1"]) # X
                
        # 直近180日間の原系列のデータ数を保存
        n_orgdata_dict[milage] = int(np.load(f"{input_path}/{milage}/n_orgdata.npz")["arr_0"])
        
        
    
    ## 逐次予測
    print("\n・予測モデル作成・逐次予測 ===============================")
    time.sleep(0.5)
    df_pred_raw = model_funcs.predWithARIMA(train_data_dict=train_data_dict, init_raw_dict=init_raw_dict, init_diff_dict=init_diff_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred, n_orgdata_dict=n_orgdata_dict)
    
        
    ## 後処理 ==================================================================================
    print("\n・後処理 ===============================")
    time.sleep(0.5)
    
    # (追加)データ読み込み
    milage_list = list(np.load(f"output/Preprocessing/track_{track}/milage_list.npz")["arr_0"])
    train_max_min = pd.Series(np.load(f"output/Preprocessing/track_{track}/train_max_min.npz")["arr_0"])
    train_max_min.index = milage_list
    
    # (変更)予測結果を検査し，異常を取得    
    abnormal_total, _ = post_funcs.diagnosePredResult(df_pred=deepcopy(df_pred_raw), train_max_min=train_max_min, tol_abnormal_max_min = tol_abnormal_max_min, tol_abnormal_upper = tol_abnormal_upper, tol_abnormal_lower = tol_abnormal_lower)
            
    # 異常値の除去
    df_pred_raw = post_funcs.postTreat(df_pred_raw=df_pred_raw, abnormal_total=abnormal_total, init_raw_dict=init_raw_dict, t_pred=t_pred)
    
    # 予測対象範囲外の除去
    df_pred_raw = df_pred_raw.iloc[range(t_pred),:]
    
    
    ## 結果保存 =============================================================
    # 予測結果を保存
    df_pred_raw.to_csv(f"output/Predicting/pred_track_{track}.csv",index=False)
    