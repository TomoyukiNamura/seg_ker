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



## 出力フォルダ作成
def makeNewFolder(folder_name):
    if os.path.exists(folder_name)==False:
        os.mkdir(folder_name)
        return True
    else:
        return False
    
output_pass = f'output/{datetime.now().strftime("%Y%m%d")}'
makeNewFolder(output_pass)


# データ読み込み
track = "C"
file_name = f"track_{track}.csv"
df_track = pd.read_csv(f"input/{file_name}")
df_track.head()


columns = "高低左"
df_irregularity = []
milage_list = list(df_track.loc[:,"キロ程"].unique())

print("行：日付id, 列：キロ程　のデータ作成")
time.sleep(0.5)
for milage in tqdm(milage_list):
    # 対象キロ程のデータを取得
    tmp_df = df_track[df_track.loc[:,"キロ程"]==milage].loc[:,[columns]]
    
    # インデックスを初期化
    tmp_df = tmp_df.reset_index(drop=True)
    
    # リネームしてアペンド
    df_irregularity.append(tmp_df.rename(columns={columns: f"m{milage}"}))

df_irregularity = pd.concat(df_irregularity, axis=1)
df_irregularity.head
df_irregularity.shape











## 設定 ===========================================================
# 予測対象キロ程，予測期間
target_milage_id_list = range(df_irregularity.shape[1])   # 原則変更しない
t_pred = 91  # 原則変更しない
start_date_id=df_irregularity.shape[0]-1 # start_date_id日目の原系列，差分系列を初期値とする＝＞start_date_id+1日目から予測

# 訓練期間
#train_date_id_list = list(range(0, 50))
#train_date_id_list.extend(list(range(100,190)))

if track=="A":
    train_date_id_list = list(range(0, 200)) # track A
    
elif track=="B":
    train_date_id_list = list(range(0, 280)) # track B
    
elif track=="C":
    #train_date_id_list = list(range(0, 220)) # track C
    train_date_id_list = list(range(280, 365)) # track C
    
elif track=="D":
    train_date_id_list = list(range(0, 250)) # track D


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

# 予測モデルの設定
model_name_pred = "lm"     #"SVR"
n_diff = 3




## 前処理 ==================================================================================
print("\n・前処理 ===============================")
time.sleep(0.5)

# 0時点の原系列，差分系列をまとめる
org_dict = {}
org_dict["raw0"] = deepcopy(df_irregularity)

# 原系列の前処理：移動平均
org_dict["raw0_prior_treated"], _, _  = make_data_funcs.priorRawData(df_raw=org_dict["raw0"], window=window, min_periods=min_periods, center=center, tol_diff=0.7, tol_n_group=5)


# 差分系列の前処理：絶対値がmu+sigma*tol_sigma超過のデータをNaNに変更
org_dict["diff0"] = make_data_funcs.priorDiffData(org_df_raw=org_dict["raw0"],df_raw=deepcopy(org_dict["raw0_prior_treated"]), n_diff=n_diff, tol_sigma=tol_sigma_diff_prior, window=window_diff, min_periods=min_periods_diff, center=center_diff)    

# n_diff+1期分の差分系列をまとめる
for i in range(n_diff):
    org_dict[f"diff{i+1}"] = org_dict["diff0"].shift(i+1)



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
df_pred_raw, start_raw_dict, maked_model_dict, inspects_dict_dict = ARIMA_funcs.predWithARIMA(org_dict=org_dict, train_dict=train_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred, model_name=model_name_pred, n_org_train_dict=n_org_train_dict)



## 後処理 ==================================================================================
print("\n・後処理 ===============================")
time.sleep(0.5)
df_pred_raw, over_tol = ARIMA_funcs.postTreat(df_pred_raw=df_pred_raw, start_raw_dict=start_raw_dict, t_pred=t_pred, tol=20)





## 結果 =============================================================
# 予測結果を保存
df_pred_raw.to_csv(f"{output_pass}/pred_{file_name}",index=False)

    

## 動画用データ保存 =================================================== 
folder_name = "pred_result_movie_honban"
makeNewFolder(folder_name)
org_dict["raw0"].to_csv(f"{folder_name}/raw0.csv",index=False,header=True)
org_dict["raw0_prior_treated"].to_csv(f"{folder_name}/raw0_prior_treated.csv",index=False,header=True)
train_dict["raw0"].to_csv(f"{folder_name}/train.csv",index=True,header=True)
df_pred_raw.to_csv(f"{folder_name}/pred_ARIMA.csv",index=True,header=True)



# cfgファイルをoutputにコピー
makeNewFolder(f"{output_pass}/{track}")
shutil.copyfile("scripts/honban.py", f"{output_pass}/{track}/honban.py")
shutil.copyfile("scripts/ARIMA_funcs.py", f"{output_pass}/{track}/ARIMA_funcs.py")
shutil.copyfile("scripts/make_data_funcs.py", f"{output_pass}/{track}/make_data_funcs.py")
