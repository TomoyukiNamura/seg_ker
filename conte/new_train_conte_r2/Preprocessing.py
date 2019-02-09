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

from func import prior_funcs


for track in ["A","B","C","D"]:
    
    print(f"\ntrack{track} ===============================")
    time.sleep(0.5)
    
    ## データ読み込み
    print(f"\n・データ読み込み ===============================")
    # 高低左データ(行：時間，列：空間)読み込み
    df_irregularity = pd.read_csv(f"input/irregularity_{track}.csv")
    
    # 補正済み高低左データ(行：時間，列：空間)読み込み
    df_irregularity_phase_modified = pd.read_csv(f"input/irregularity_{track}_phase_modified.csv")

    
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
    n_diff = config.getint('model', 'n_diff')
    train_date_id_list = list(range(config.getint('model', 'train_date_id_start'), config.getint('model', 'train_date_id_end')))
    
    
    
    ## 予測対象キロ程，予測期間の設定
    start_date_id=df_irregularity_phase_modified.shape[0] -1 # start_date_id日目の原系列，差分系列を初期値とする＝＞start_date_id+1日目から予測
    t_pred = 91  # 変更しない
    
    
    
    ## あとで消す ===================
    target_milage_id_list = range(50)
    df_irregularity = df_irregularity.iloc[:,target_milage_id_list]
    df_irregularity_phase_modified = df_irregularity_phase_modified.iloc[:,target_milage_id_list]
    ## あとで消す ===================
    
    
    
    ## 前処理 ==================================================================================
    print("\n・前処理 ===============================")
    time.sleep(0.5)
    
    # 0時点の原系列，差分系列をまとめる
    org_dict = {}
    org_dict["raw0"] = deepcopy(df_irregularity_phase_modified)
    
    # 原系列の前処理：移動平均
    org_dict["raw0_prior_treated"], _, _  = prior_funcs.priorRawData(df_raw=org_dict["raw0"], window=window, min_periods=min_periods, center=center, tol_diff=0.7, tol_n_group=5, window_median=20)
    
    
    # 差分系列の前処理：絶対値がmu+sigma*tol_sigma超過のデータをNaNに変更
    org_dict["diff0"] = prior_funcs.priorDiffData(org_df_raw=org_dict["raw0"],df_raw=deepcopy(org_dict["raw0_prior_treated"]), n_diff=n_diff, tol_sigma=tol_sigma_diff_prior, window=window_diff, min_periods=min_periods_diff, center=center_diff)    
    
    # n_diff+1期分の差分系列をまとめる
    for i in range(n_diff):
        org_dict[f"diff{i+1}"] = org_dict["diff0"].shift(i+1)
    
    
    
    ## 初期値の準備
    print("\n原系列初期値を取得")
    time.sleep(0.5)
    init_raw_dict = prior_funcs.makeStartRawDict(df_raw=df_irregularity, start_date_id=start_date_id, start_period=start_period, n_average_date=n_average_date, start_average_method=start_average_method)
    
    
    print("\n差分系列初期値を取得")
    time.sleep(0.5)
    init_diff_dict, _ = prior_funcs.makeStartDiffDict(df_dict=org_dict, n_diff=n_diff, start_date_id=start_date_id)

    
    # 訓練データを取得 
    train_dict = {}
    for key in list(org_dict.keys()):
        train_dict[key] = deepcopy(org_dict[key].loc[train_date_id_list,:])
        
    
    # オリジナル原系列の訓練データ範囲のデータ数調査
    print("\n・直近180日間の原系列にどれほどデータ数があるか===============================")
    time.sleep(0.5)
    n_org_train_dict = {}
    n_org = org_dict["raw0"].shape[0]
    tmp_bool = org_dict["raw0"].iloc[list(range(n_org-180, n_org)),:].isna()==False
    for milage in tqdm(list(org_dict["raw0"].columns)):
        n_org_train_dict[milage] = tmp_bool[milage][tmp_bool[milage]==True].shape[0]
    
    
    # (追加)モデルに入力可能な形式にデータを変換
    print("\n・モデルに入力可能な形式にデータを変換===============================")
    time.sleep(0.5)
    training_data_dict = {}
    for milage in tqdm(list(train_dict["raw0"].columns)):
        # y：目的変数(t時点差分),X：説明変数(t-1, t-2, t-3時点差分)
        training_data_dict[milage] = prior_funcs.dfDict2ARIMAInput(train_dict, n_diff, milage)
        
        
    
    # (追加)訓練データ，初期値を保存
    # キロ程リスト
    np.savez(f"output/Preprocessing/track_{track}/milage_list.npz", list(df_irregularity.columns))
    
    # 
    train_max_min = train_dict["raw0_prior_treated"].max() - train_dict["raw0_prior_treated"].min()
    np.savez(f"output/Preprocessing/track_{track}/train_max_min.npz", train_max_min)
    
    # 
    print("\n・モデルに入力可能な形式にデータを変換===============================")
    time.sleep(0.5)
    for milage in tqdm(list(df_irregularity.columns)):
        # アウトプットフォルダを作成
        output_path = f"output/Preprocessing/track_{track}/{milage}"
        if os.path.exists(output_path)==False: os.mkdir(output_path)
        
        # 初期原データを保存
        np.savez(f"{output_path}/init_raw.npz", init_raw_dict[milage])
        
        # 初期差分を保存
        np.savez(f"{output_path}/init_diff.npz", init_diff_dict[milage])
        
        # 訓練データを保存
        np.savez(f"{output_path}/train_data.npz", training_data_dict[milage][0], training_data_dict[milage][1])
        
        # 直近180日間の原系列のデータ数を保存
        np.savez(f"{output_path}/n_orgdata.npz", n_org_train_dict[milage])
        
        
        
        
        
        
    
    