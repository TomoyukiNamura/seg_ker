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

from scripts import make_data_funcs
from scripts import ARIMA_funcs


# データ読み込み
track = "A"
df_irregularity = pd.read_csv(f"input/irregularity_{track}.csv")
df_irregularity_phase_modified = pd.read_csv(f"input/irregularity_{track}_phase_modified.csv")

df_irregularity_phase_modified.head()
df_irregularity_phase_modified.tail()
df_irregularity_phase_modified.shape




## 出力フォルダ作成
output_pass = f'output/{datetime.now().strftime("%Y%m%d")}'
make_data_funcs.makeNewFolder(output_pass)


## 設定 ===========================================================
# 訓練期間
if track=="A":
    train_date_id_list = list(range(250, 365)) # track A
    lag_t = 0
    tol_abnormal_upper = 13
    tol_abnormal_lower = -13
    method_post = "mean"
    
elif track=="B":
    train_date_id_list = list(range(100,280))
    lag_t = 12
    tol_abnormal_upper = 17
    tol_abnormal_lower = -17
    method_post = "aaa"
    
elif track=="C":
    train_date_id_list = list(range(280, 365)) # track C
    lag_t = 0
    tol_abnormal_upper = 20
    tol_abnormal_lower = -20
    method_post = "aaa"

elif track=="D":
    train_date_id_list = list(range(140, 240))
    lag_t = 0
    tol_abnormal_upper = 17
    tol_abnormal_lower = -17
    method_post = "aaa"


# 予測対象キロ程，予測期間
target_milage_id_list = range(df_irregularity_phase_modified.shape[1])   # 変更しない
t_pred = 91  # 変更しない
start_date_id=df_irregularity_phase_modified.shape[0] -1 -lag_t # start_date_id日目の原系列，差分系列を初期値とする＝＞start_date_id+1日目から予測


# 前処理(原系列)の設定
tol_sigma_raw_prior = 2.5
window=50
min_periods=3
center=True

# 前処理(差分系列)の設定
tol_sigma_diff_prior = 2.0
window_diff=3
min_periods_diff=1
center_diff=True

# 前処理(初期値)の設定
start_period = 30
n_average_date = 5
start_average_method = "median"#"median", "mean"

# 予測モデルの設定
model_name_pred = "lm"     # "lm", "SVR", "mean"
n_diff = 3


## 後処理(予測結果修正)の設定
tol_abnormal_max_min = 2.5



## 前処理 ==================================================================================
print("\n・前処理 ===============================")
time.sleep(0.5)

# 0時点の原系列，差分系列をまとめる
org_dict = {}
org_dict["raw0"] = deepcopy(df_irregularity_phase_modified)

# 原系列の前処理：移動平均
org_dict["raw0_prior_treated"], _, _  = make_data_funcs.priorRawData(df_raw=org_dict["raw0"], window=window, min_periods=min_periods, center=center, tol_diff=0.7, tol_n_group=5)


# 差分系列の前処理：絶対値がmu+sigma*tol_sigma超過のデータをNaNに変更
org_dict["diff0"] = make_data_funcs.priorDiffData(org_df_raw=org_dict["raw0"],df_raw=deepcopy(org_dict["raw0_prior_treated"]), n_diff=n_diff, tol_sigma=tol_sigma_diff_prior, window=window_diff, min_periods=min_periods_diff, center=center_diff)    

# n_diff+1期分の差分系列をまとめる
for i in range(n_diff):
    org_dict[f"diff{i+1}"] = org_dict["diff0"].shift(i+1)



## 初期値の準備
print("\n原系列初期値を取得")
time.sleep(0.5)
start_raw_dict = make_data_funcs.makeStartRawDict(df_raw=df_irregularity, start_date_id=start_date_id, start_period=start_period, n_average_date=n_average_date, start_average_method=start_average_method)


print("\n差分系列初期値を取得")
time.sleep(0.5)
start_diff_dict, start_values_result_dict = make_data_funcs.makeStartDiffDict(df_dict=org_dict, n_diff=n_diff, start_date_id=start_date_id)



## 予測モデル作成・逐次予測===============================================================
# 訓練データを取得 
train_dict = {}
for key in list(org_dict.keys()):
    train_dict[key] = deepcopy(org_dict[key].iloc[train_date_id_list,:])
 

# オリジナル原系列の訓練データ範囲のデータ数調査
print("\n・オリジナル原系列の訓練データ範囲のデータ数調査===============================")
time.sleep(0.5)
n_org_train_dict = {}
tmp_bool = org_dict["raw0"].iloc[train_date_id_list,:].isna()==False
for milage in tqdm(list(org_dict["raw0"].columns)):
    n_org_train_dict[milage] = tmp_bool[milage][tmp_bool[milage]==True].shape[0]


## 逐次予測
print("\n・予測モデル作成・逐次予測 ===============================")
time.sleep(0.5)
df_pred_raw = ARIMA_funcs.predWithARIMA(train_dict=train_dict, start_raw_dict=start_raw_dict, start_diff_dict=start_diff_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred+lag_t, model_name=model_name_pred, n_org_train_dict=n_org_train_dict)


# 絶対値20越えの結果を保存
tmp = (np.abs(df_pred_raw) > 20).any(axis=0)
len(tmp[tmp])
df_pred_raw.loc[:,tmp]

folder_name = "pred_result_movie_honban"
make_data_funcs.makeNewFolder(folder_name)
train_dict["raw0"].loc[:,tmp].to_csv(f"{folder_name}/train_over_tol.csv",index=True,header=True)
df_pred_raw.loc[:,tmp].to_csv(f"{folder_name}/pred_ARIMA_over_tol.csv",index=True,header=True)



## 後処理 ==================================================================================
print("\n・後処理 ===============================")
time.sleep(0.5)
abnormal_total, diagnosis_result = ARIMA_funcs.diagnosePredResult(df_pred=deepcopy(df_pred_raw), df_train=deepcopy(train_dict["raw0_prior_treated"]), tol_abnormal_max_min = tol_abnormal_max_min, tol_abnormal_upper = tol_abnormal_upper, tol_abnormal_lower = tol_abnormal_lower)

folder_name = "pred_result_movie_honban"
make_data_funcs.makeNewFolder(folder_name)
train_dict["raw0"].loc[:,abnormal_total].to_csv(f"{folder_name}/train_over_tol.csv",index=True,header=True)
df_pred_raw.loc[:,abnormal_total].to_csv(f"{folder_name}/pred_ARIMA_over_tol.csv",index=True,header=True)

df_pred_raw = ARIMA_funcs.postTreat(df_pred_raw=df_pred_raw, abnormal_total=abnormal_total, start_raw_dict=start_raw_dict, t_pred=t_pred+lag_t, method=method_post)
df_pred_raw = df_pred_raw.iloc[range(lag_t, t_pred+lag_t),:]


## 結果 =============================================================
# 予測結果を保存
df_pred_raw.to_csv(f"{output_pass}/pred_track_{track}.csv",index=False)

# cfgファイルをoutputにコピー
make_data_funcs.makeNewFolder(f"{output_pass}/{track}")
shutil.copyfile("scripts/honban.py", f"{output_pass}/{track}/honban.py")
shutil.copyfile("scripts/ARIMA_funcs.py", f"{output_pass}/{track}/ARIMA_funcs.py")
shutil.copyfile("scripts/make_data_funcs.py", f"{output_pass}/{track}/make_data_funcs.py")



## 動画用データ保存 =================================================== 
folder_name = "pred_result_movie_honban"
make_data_funcs.makeNewFolder(folder_name)
shutil.copyfile("input/irregularity_A.csv", f"{folder_name}/org_raw0.csv")
org_dict["raw0"].to_csv(f"{folder_name}/raw0.csv",index=False,header=True)
org_dict["raw0_prior_treated"].to_csv(f"{folder_name}/raw0_prior_treated.csv",index=False,header=True)
train_dict["raw0"].to_csv(f"{folder_name}/train.csv",index=True,header=True)
train_dict["raw0_prior_treated"].to_csv(f"{folder_name}/train_prior_treated.csv",index=True,header=True)
df_pred_raw.to_csv(f"{folder_name}/pred_ARIMA.csv",index=True,header=True)



#tmp = (np.abs(df_pred_raw) > 10).any(axis=0)
#df_pred_raw.loc[:,tmp].to_csv(f"{folder_name}/pred_ARIMA.csv",index=True,header=True)
#df_pred_raw.loc[:,tmp].shape