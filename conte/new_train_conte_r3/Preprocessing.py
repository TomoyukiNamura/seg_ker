#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Preprocessing

"""


import os
os.chdir("/Users/tomoyuki/python_workspace/new_train_conte")

import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import time
import configparser

from func import preprocess_func


for track in ["A","B","C","D"]:
    
    print(f"\ntrack{track} ===============================")
    time.sleep(0.5)
        
    ## 初期処理 
    ## 設定ファイル読み込み
    config = configparser.ConfigParser()
    config.read(f"conf/main.ini", 'UTF-8')
    
    tol_sigma_raw_prior = config.getfloat('preprocess', 'tol_sigma_raw_prior')
    window = config.getint('preprocess', 'window')
    min_periods = config.getint('preprocess', 'min_periods')
    center = config.getboolean('preprocess', 'center')
    tol_sigma_diff_prior = config.getfloat('preprocess', 'tol_sigma_diff_prior')
    window_diff = config.getint('preprocess', 'window_diff')
    min_periods_diff = config.getint('preprocess', 'min_periods_diff')
    center_diff = config.getboolean('preprocess', 'center_diff')
    start_period = config.getint('preprocess', 'start_period')
    n_average_date = config.getint('preprocess', 'n_average_date')
    start_average_method = config.get('preprocess', 'start_average_method')
    n_diff = config.getint('preprocess', 'n_diff')
    train_date_id_list = list(range(config.getint('preprocess', f'train_date_id_start_track_{track}'), config.getint('preprocess', f'train_date_id_end_track_{track}')))
    
    
    # 元データを読み込み，「行：日付id, 列：キロ程」の高低左データに変換
    #df_irregularity = preprocess_func.makeTimeSpaceData(f"input/track_{track}.csv")
    df_irregularity = pd.read_csv(f"input/irregularity_{track}.csv")
    
    
    ## 予測対象キロ程，予測期間の設定
    start_date_id = df_irregularity.shape[0] -1 # start_date_id日目の原系列，差分系列を初期値とする＝＞start_date_id+1日目から予測    

    # 原データ初期値を取得
    init_raw_dict = preprocess_func.makeStartRawDict(df_raw=df_irregularity, start_date_id=start_date_id, start_period=start_period, n_average_date=n_average_date, start_average_method=start_average_method)
    
    
    # 位相補正
#    df_irregularity_phase_modified = phaseModify(df_irregularity ,n_shift, n_sub_data)
    df_irregularity_phase_modified = pd.read_csv(f"input/irregularity_{track}_phase_modified.csv")


    ## あとで消す ===================
    target_milage_id_list = range(50)
    df_irregularity = df_irregularity.iloc[:,target_milage_id_list]
    df_irregularity_phase_modified = df_irregularity_phase_modified.iloc[:,target_milage_id_list]
    ## あとで消す ===================

    
    # オリジナル原系列の訓練データ範囲のデータ数調査
    n_org_train_dict = preprocess_func.checkOrgTrainNum(df_irregularity_phase_modified)


    # 0時点の原系列，差分系列をまとめる
    org_dict = {}
    org_dict["raw0"] = deepcopy(df_irregularity_phase_modified)
    
    # 原系列の前処理：移動平均
    org_dict["raw0_prior_treated"], _, _  = preprocess_func.priorRawData(df_raw=org_dict["raw0"], window=window, min_periods=min_periods, center=center, tol_diff=0.7, tol_n_group=5, window_median=20)
    
    # 差分系列の前処理：絶対値がmu+sigma*tol_sigma超過のデータをNaNに変更
    org_dict["diff0"] = preprocess_func.priorDiffData(org_df_raw=org_dict["raw0"],df_raw=deepcopy(org_dict["raw0_prior_treated"]), n_diff=n_diff, tol_sigma=tol_sigma_diff_prior, window=window_diff, min_periods=min_periods_diff, center=center_diff)    
    
    # n_diff+1期分の差分系列をまとめる
    for i in range(n_diff):
        org_dict[f"diff{i+1}"] = org_dict["diff0"].shift(i+1)
    
    # 差分初期値を取得
    init_diff_dict, _ = preprocess_func.makeStartDiffDict(df_dict=org_dict, n_diff=n_diff, start_date_id=start_date_id)

    # 訓練期間以外のデータを除外 
    for key in list(org_dict.keys()):
        org_dict[key] = deepcopy(org_dict[key].loc[train_date_id_list,:])
   
    
    # (追加)モデルに入力可能な形式にデータを変換(y：目的変数(t時点差分),X：説明変数(t-1, t-2, t-3時点差分))
    training_data_dict = preprocess_func.dfDict2ARIMAInput(org_dict, n_diff)
    
    
    # (追加)訓練データ，初期値を保存
    milage_list = list(df_irregularity_phase_modified.columns)
    train_max_min = org_dict["raw0_prior_treated"].max() - org_dict["raw0_prior_treated"].min()
    preprocess_func.savePreprocessData(milage_list, train_max_min, init_raw_dict, init_diff_dict, training_data_dict, n_org_train_dict, f"output/Preprocessing/track_{track}")
    
    