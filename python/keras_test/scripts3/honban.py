#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir("/Users/tomoyuki/Desktop/keras_test")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import time
import shutil
import configparser

from scripts.func import prior_funcs
from scripts.func import model_funcs
from scripts.func import post_funcs

# 出力フォルダ作成関数
def makeNewFolder(folder_name):
    if os.path.exists(folder_name)==False:
        os.mkdir(folder_name)
        return True
    else:
        return False
    


# 設定
track_list = ["C"]

output_movie = True


## 出力フォルダ作成 ===================================================================================
output_pass = f'output/{datetime.now().strftime("%Y%m%d")}'
makeNewFolder(output_pass)


for track in track_list:
    
    print(f"\ntrack{track} ===============================")
    time.sleep(0.5)
    
    if output_movie:
        # 動画用フォルダ作成
        movie_folder_name = f"output/0_track{track}_movie_pred_result_honban"
        makeNewFolder(movie_folder_name)
        
    ## データ読み込み ===================================================================================
    print(f"\n・データ読み込み ===============================")
    time.sleep(0.5)
    df_irregularity = pd.read_csv(f"input/irregularity_{track}.csv")
    df_irregularity_phase_modified = pd.read_csv(f"input/irregularity_{track}_phase_modified.csv")
    
    
    
    ## 初期処理 ===================================================================================
    ## 設定ファイル読み込み
    config = configparser.ConfigParser()
    config.read(f"scripts/conf/track_{track}.ini", 'UTF-8')
    
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
    
    # [others]
    lag_t = config.getint('others', 'lag_t')
    
    
    # nnet設定
    n_state = 1
    stride = n_state*2 + 1 # 重複無し
    #stride = 1             # 重複あり
    tol_n_raw = 30
    batch_size = 10
    epochs = 20
    
    
    ## 予測対象キロ程，予測期間の設定
    target_milage_id_list = range(df_irregularity_phase_modified.shape[1])   # 変更しない
    t_pred = 91  # 変更しない
    start_date_id=df_irregularity_phase_modified.shape[0] -1 -lag_t # start_date_id日目の原系列，差分系列を初期値とする＝＞start_date_id+1日目から予測
    
    
    
    
    
#    target_milage_id_list = range(2200,2400)
    
#    org_df_irregularity = deepcopy(df_irregularity)
#    org_df_irregularity_phase_modified = deepcopy(df_irregularity_phase_modified)
    
#    df_irregularity = deepcopy(org_df_irregularity.iloc[:,target_milage_id_list])
#    df_irregularity_phase_modified = deepcopy(org_df_irregularity_phase_modified.iloc[:,target_milage_id_list])
        
    
    ## 前処理 ==================================================================================
    print("\n・前処理 ===============================")
    time.sleep(0.5)
    
    # 0時点の原系列，差分系列をまとめる
    org_dict = {}
    org_dict["raw0"] = deepcopy(df_irregularity_phase_modified)
    
    # 原系列の前処理：移動平均
    org_dict["raw0_prior_treated"], _, _  = prior_funcs.priorRawData(df_raw=org_dict["raw0"], window=window, min_periods=min_periods, center=center, tol_diff=0.7, tol_n_group=5, window_median=10)
    
    
    # 差分系列の前処理：絶対値がmu+sigma*tol_sigma超過のデータをNaNに変更
    org_dict["diff0"] = prior_funcs.priorDiffData(org_df_raw=org_dict["raw0"],df_raw=deepcopy(org_dict["raw0_prior_treated"]), n_diff=n_diff, tol_sigma=tol_sigma_diff_prior, window=window_diff, min_periods=min_periods_diff, center=center_diff)    
    
    # n_diff+1期分の差分系列をまとめる
    for i in range(n_diff):
        org_dict[f"diff{i+1}"] = org_dict["diff0"].shift(i+1)
    
    
    
    ## 初期値の準備
    print("\n原系列初期値を取得")
    time.sleep(0.5)
    start_raw_dict = prior_funcs.makeStartRawDict(df_raw=df_irregularity, start_date_id=start_date_id, start_period=start_period, n_average_date=n_average_date, start_average_method=start_average_method)
    
    
    print("\n差分系列初期値を取得")
    time.sleep(0.5)
    start_diff_dict, start_values_result_dict = prior_funcs.makeStartDiffDict(df_dict=org_dict, n_diff=n_diff, start_date_id=start_date_id)
    
    
    
    
    
    ## 予測モデル作成・逐次予測===============================================================
    # 訓練データを取得 
    train_dict = {}
    for key in list(org_dict.keys()):
        train_dict[key] = deepcopy(org_dict[key].loc[train_date_id_list,:])
        
    
    # オリジナル原系列の訓練データ範囲のデータ数調査
    print("\n・直近150日間の原系列にどれほどデータ数があるか===============================")
    time.sleep(0.5)
    n_org_train_dict = {}
    n_org = org_dict["raw0"].shape[0]
    tmp_bool = org_dict["raw0"].iloc[list(range(n_org-180, n_org)),:].isna()==False
    for milage in tqdm(list(org_dict["raw0"].columns)):
        n_org_train_dict[milage] = tmp_bool[milage][tmp_bool[milage]==True].shape[0]
    
    
    ## 逐次予測
    print("\n・予測モデル作成・逐次予測 ===============================")
    time.sleep(0.5)
    
    if model_name_pred == "nnet":
        milage_list_list = model_funcs.makeMilageListList(n_state=n_state, stride=stride, tol_n_raw=tol_n_raw, n_org_train_dict=n_org_train_dict)
        df_pred_raw = model_funcs.predWithSpatialAriNnet(milage_list_list=milage_list_list, train_dict=train_dict, start_raw_dict=start_raw_dict, start_diff_dict=start_diff_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred+lag_t, batch_size=batch_size, epochs=epochs)
    
    else:
        df_pred_raw = model_funcs.predWithARIMA(train_dict=train_dict, start_raw_dict=start_raw_dict, start_diff_dict=start_diff_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred+lag_t, model_name=model_name_pred, n_org_train_dict=n_org_train_dict)
        
    
    
    ## 後処理 ==================================================================================
    print("\n・後処理 ===============================")
    time.sleep(0.5)
    
    # 予測結果を検査し，異常を取得
    abnormal_total, diagnosis_result = post_funcs.diagnosePredResult(df_pred=deepcopy(df_pred_raw), df_train=deepcopy(train_dict["raw0_prior_treated"]), tol_abnormal_max_min = tol_abnormal_max_min, tol_abnormal_upper = tol_abnormal_upper, tol_abnormal_lower = tol_abnormal_lower)
    
    # 異常あり結果を出力
    if output_movie:
        train_dict["raw0"].loc[:,abnormal_total].to_csv(f"{movie_folder_name}/train_over_tol.csv",index=True,header=True)
        df_pred_raw.loc[:,abnormal_total].to_csv(f"{movie_folder_name}/pred_ARIMA_over_tol.csv",index=True,header=True)
        
    # 異常値の除去
    df_pred_raw = post_funcs.postTreat(df_pred_raw=df_pred_raw, abnormal_total=abnormal_total, start_raw_dict=start_raw_dict, t_pred=t_pred+lag_t, method=method_post)
    
    # 予測対象範囲外の除去
    df_pred_raw = df_pred_raw.iloc[range(lag_t, t_pred+lag_t),:]
    
    
    ## 結果保存 =============================================================
    # 予測結果を保存
    df_pred_raw.to_csv(f"{output_pass}/pred_track_{track}.csv",index=False)
    
    # スクリプト
    makeNewFolder(f"{output_pass}/scripts")
    shutil.copyfile("scripts/honban.py", f"{output_pass}/scripts/honban.py")
    shutil.copyfile("scripts/finalize_result.py", f"{output_pass}/scripts/finalize_result.py")
    
    # ファンクション
    makeNewFolder(f"{output_pass}/scripts/func")
    shutil.copyfile("scripts/func/prior_funcs.py", f"{output_pass}/scripts/func/prior_funcs.py")
    shutil.copyfile("scripts/func/model_funcs.py", f"{output_pass}/scripts/func/model_funcs.py")
    shutil.copyfile("scripts/func/nnet_model.py", f"{output_pass}/scripts/func/nnet_model.py")
    shutil.copyfile("scripts/func/post_funcs.py", f"{output_pass}/scripts/func/post_funcs.py")
    
    # コンフィグ
    makeNewFolder(f"{output_pass}/scripts/conf")
    shutil.copyfile(f"scripts/conf/track_{track}.ini", f"{output_pass}/scripts/conf/track_{track}.ini")
    
    
    if output_movie:
        ## 動画用データ保存 =================================================== 
        shutil.copyfile(f"input/irregularity_{track}.csv", f"{movie_folder_name}/org_raw0.csv")
        org_dict["raw0"].iloc[:,range(len(target_milage_id_list))].to_csv(f"{movie_folder_name}/raw0.csv",index=False,header=True)
        org_dict["raw0_prior_treated"].iloc[:,range(len(target_milage_id_list))].to_csv(f"{movie_folder_name}/raw0_prior_treated.csv",index=False,header=True)
        train_dict["raw0"].iloc[:,range(len(target_milage_id_list))].to_csv(f"{movie_folder_name}/train.csv",index=True,header=True)
        train_dict["raw0_prior_treated"].iloc[:,range(len(target_milage_id_list))].to_csv(f"{movie_folder_name}/train_prior_treated.csv",index=True,header=True)
        df_pred_raw.to_csv(f"{movie_folder_name}/pred_ARIMA.csv",index=True,header=True)
        
    
    
    
    
    time.sleep(5)



#tmp = (np.abs(df_pred_raw) > 10).any(axis=0)
#df_pred_raw.loc[:,tmp].to_csv(f"{folder_name}/pred_ARIMA.csv",index=True,header=True)
#df_pred_raw.loc[:,tmp].shape


#np.max(np.max(df_pred_raw))
#np.min(np.min(df_pred_raw))
