#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir("/Users/tomoyuki/Desktop/keras_test")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import time
import shutil
from sklearn.linear_model import LinearRegression as lm
from sklearn.neighbors import KNeighborsRegressor as knn

from scripts import make_data_funcs
from scripts import ARIMA_funcs

import scripts.model as model


# データ読み込み
track = "A"
df_irregularity = pd.read_csv(f"input/irregularity_{track}.csv")
df_irregularity_phase_modified = pd.read_csv(f"input/irregularity_{track}_phase_modified.csv")

df_irregularity_phase_modified.head()
df_irregularity_phase_modified.tail()
df_irregularity_phase_modified.shape




## スモールデータ =====================================
# 訓練データ，評価データの設定
#target_milage_id_list = range(6935,6945)
#target_milage_id_list = range(6900,6970)
#target_milage_id_list = range(6800,7200)
#target_milage_id_list = range(6970,7100)
#target_milage_id_list = range(1700,1730)#1723
target_milage_id_list = range(8580,8600)#1723


if track=="A":
    t_pred = 41#91
    start_date_id=150  # start_date_id日目の原系列，差分系列を初期値とする＝＞start_date_id+1日目から予測
    train_date_id_list = list(range(0, 200)) # track A
    lag_t = 0
    start_raw_file_name = f"irregularity_{track}.csv"
    
elif track=="B":
    t_pred = 41#91
    start_date_id=100
    #train_date_id_list = list(range(0, 280)) # track B
    #train_date_id_list = list(range(300,360))
    train_date_id_list = list(range(0,100))
    #train_date_id_list = list(range(150, 280))
    lag_t = 12
    t_pred = 91
    start_raw_file_name = f"irregularity_{track}.csv"
    
    
elif track=="C":
    t_pred = 41#91
    start_date_id=150
    #train_date_id_list = list(range(0, 220)) # track C
    train_date_id_list = list(range(280, 365)) # track C
    lag_t = 0
    start_raw_file_name = f"irregularity_{track}.csv"
    
elif track=="D":
    t_pred = 41#91
    start_date_id=10
#    train_date_id_list = list(range(0, 250)) # track D
    train_date_id_list = list(range(140, 240))
    lag_t = 0
    start_raw_file_name = f"irregularity_{track}.csv"

start_date_id = start_date_id -1 - lag_t
test_date_id_list = range(start_date_id+1+lag_t, start_date_id+1+lag_t+t_pred)



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
start_average_method = "mean"#"median"


# 予測モデルの設定
model_name_pred = "lm"     #"SVR"
n_diff = 3

## 後処理(予測結果修正)の設定
tol_abnormal_max_min = 2.5
tol_abnormal_upper = 25
tol_abnormal_lower = -25


## スモールデータ取得=========================================
df_irregularity_small = deepcopy(df_irregularity.iloc[:,target_milage_id_list])
df_irregularity_phase_modified_small = deepcopy(df_irregularity_phase_modified.iloc[:,target_milage_id_list])



## 8700 ~見直し

## 前処理 ==================================================================================
print("\n・前処理 ===============================")
time.sleep(0.5)

# 0時点の原系列，差分系列をまとめる
org_dict = {}
org_dict["raw0"] = deepcopy(df_irregularity_phase_modified_small)

# 原系列の前処理：移動平均
org_dict["raw0_prior_treated"], tmp_raw0_median, tmp_raw0_median_diff  = make_data_funcs.priorRawData(df_raw=org_dict["raw0"], window=window, min_periods=min_periods, center=center, tol_diff=0.7, tol_n_group=5)


folder_name = "movie"
make_data_funcs.makeNewFolder(folder_name)
org_dict["raw0"].to_csv(f"{folder_name}/raw0.csv",index=False,header=True)
tmp_raw0_median.to_csv(f"{folder_name}/tmp_raw0_median.csv",index=False,header=True)
tmp_raw0_median_diff.to_csv(f"{folder_name}/diff_prior_treated.csv",index=False,header=True)
org_dict["raw0_prior_treated"].to_csv(f"{folder_name}/raw0_prior_treated.csv",index=False,header=True)




# 差分系列の前処理：絶対値がmu+sigma*tol_sigma超過のデータをNaNに変更
org_dict["diff0"] = make_data_funcs.priorDiffData(org_df_raw=org_dict["raw0"], df_raw=deepcopy(org_dict["raw0_prior_treated"]), n_diff=n_diff, tol_sigma=tol_sigma_diff_prior, window=window_diff, min_periods=min_periods_diff, center=center_diff)
    

# n_diff+1期分の差分系列をまとめる
for i in range(n_diff):
    org_dict[f"diff{i+1}"] = org_dict["diff0"].shift(i+1)



## 初期値の準備
print("\n原系列初期値を取得")
time.sleep(0.5)
start_raw_dict = make_data_funcs.makeStartRawDict(df_raw=df_irregularity_small, start_date_id=start_date_id, start_period=start_period, n_average_date=n_average_date, start_average_method=start_average_method)


print("\n差分系列初期値を取得")
time.sleep(0.5)
start_diff_dict, start_values_result_dict = make_data_funcs.makeStartDiffDict(df_dict=org_dict, n_diff=n_diff, start_date_id=start_date_id)





## 予測モデル作成・逐次予測===============================================================
# 訓練データを取得 
train_dict = {}
for key in list(org_dict.keys()):
    train_dict[key] = deepcopy(org_dict[key].iloc[train_date_id_list,:])
    

print("\n・オリジナル原系列の訓練データ範囲のデータ数調査===============================")
time.sleep(0.5)
n_org_train_dict = {}
for milage in tqdm(list(org_dict["raw0"].columns)):
    n_org_train_dict[milage] = org_dict["raw0"].iloc[train_date_id_list,:][milage].dropna().shape[0]


## ARIMA(n_diff,1,0)による逐次予測
print("\n・予測モデル作成・逐次予測 ===============================")
print("ARIMA")
time.sleep(0.5)
df_pred_raw_lm = ARIMA_funcs.predWithARIMA(train_dict=train_dict, start_raw_dict=start_raw_dict, start_diff_dict=start_diff_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred+lag_t, model_name="lm", n_org_train_dict=n_org_train_dict)


print("直近5日間中央値")
time.sleep(0.5)
df_pred_raw_mean = ARIMA_funcs.predWithARIMA(train_dict=train_dict, start_raw_dict=start_raw_dict, start_diff_dict=start_diff_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred+lag_t, model_name="median", n_org_train_dict=n_org_train_dict)


## 後処理 ==================================================================================
print("\n・後処理 ===============================")
time.sleep(0.5)
abnormal_total, diagnosis_result = ARIMA_funcs.diagnosePredResult(df_pred=deepcopy(df_pred_raw_lm), df_train=deepcopy(train_dict["raw0_prior_treated"]), tol_abnormal_max_min = tol_abnormal_max_min, tol_abnormal_upper = tol_abnormal_upper, tol_abnormal_lower = tol_abnormal_lower)
df_pred_raw_lm = ARIMA_funcs.postTreat(df_pred_raw=df_pred_raw_lm, abnormal_total=abnormal_total, start_raw_dict=start_raw_dict, t_pred=t_pred+lag_t)
df_pred_raw_lm = df_pred_raw_lm.iloc[range(lag_t, t_pred+lag_t),:]
df_pred_raw_mean = df_pred_raw_mean.iloc[range(lag_t, t_pred+lag_t),:]



## 結果 =============================================================
# 評価データを取得 
test_dict = {}
for key in list(org_dict.keys()):
    test_dict[key] = deepcopy(org_dict[key].iloc[test_date_id_list,:])

# MAE計算・プロット
print("MAE lm")
mae_dict_lm = {}
for milage in list(df_pred_raw_lm.columns):
    mae_dict_lm[milage] = ARIMA_funcs.calcMAE(df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_lm[milage])
ARIMA_funcs.plotTotalMAE(mae_dict=mae_dict_lm, ylim=[0.0, 1.0], r_plot_size=1, output_dir=f"ARIMA_{{track}}_lm")

print("MAE mean")
mae_dict_mean = {}
for milage in list(df_pred_raw_mean.columns):
    mae_dict_mean[milage] = ARIMA_funcs.calcMAE(df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_mean[milage])
ARIMA_funcs.plotTotalMAE(mae_dict=mae_dict_mean, ylim=[0.0, 1.0], r_plot_size=1, output_dir=f"ARIMA_{{track}}_mean")


# lmとmeanのMAEの差分を計算・プロット(-の値の部分でlmが優っている)
diff_mae = np.array(list(mae_dict_lm.values())) - np.array(list(mae_dict_mean.values()))
plt.plot(diff_mae, color="red");plt.ylim([-0.1, 0.1]);plt.grid();plt.show()



folder_name = "pred_result_movie"
make_data_funcs.makeNewFolder(folder_name)

train_dict["raw0"].to_csv(f"{folder_name}/train.csv",index=True,header=True)
test_dict["raw0"].to_csv(f"{folder_name}/test.csv",index=True,header=True)
df_pred_raw_lm.to_csv(f"{folder_name}/pred_ARIMA.csv",index=True,header=True)
df_pred_raw_mean.to_csv(f"{folder_name}/pred_mean.csv",index=True,header=True)

pd.DataFrame({"milage":list(mae_dict_lm.keys()), "MAE":list(mae_dict_lm.values())}).to_csv(f"{folder_name}/MAE_ARIMA.csv",index=False,header=True)
pd.DataFrame({"milage":list(mae_dict_mean.keys()), "MAE":list(mae_dict_mean.values())}).to_csv(f"{folder_name}/MAE_mean.csv",index=False,header=True)





# 結果をプロット
milage_id_list = range(0,1)
ylim=[-10,10]
for milage_id in milage_id_list:
    milage = list(df_pred_raw_lm.columns)[milage_id]
    print(f"\n{milage_id} {milage} ======================================================\n")
    print("lm")
        
    ARIMA_funcs.PlotTruthPred(df_train=train_dict["raw0"][milage], df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_lm[milage], inspects_dict=None, 
            ylim=ylim, r_plot_size=1,output_dir=f"ARIMA_{track}_lm", file_name=f"{milage_id}_{milage}")
    
    print("mean")
    ARIMA_funcs.PlotTruthPred(df_train=train_dict["raw0"][milage], df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_mean[milage], inspects_dict=None, 
            ylim=ylim, r_plot_size=1,output_dir=f"ARIMA_{track}_mean", file_name=f"{milage_id}_{milage}")
        
    



    
### ニューラルネットワーク ========================================
## dfから入力データ作成
#y, X = model.dfDict2SAMInput(df_diff=df_spatio_diff)
#
## spatialARIモデル作成
#spatialARIModel = model.spatialARIModel(input_shape=(X.shape[1],X.shape[2]))
#spatialARIModel.summary()
#
## 学習
#model.fit(x=X, y=y, batch_size=10, epochs=10, verbose=1)
#model.get_weights()
#
#


