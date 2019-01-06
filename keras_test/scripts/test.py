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
from scripts.func import test_funcs


## データ読み込み ===================================================================================
track = "A"
df_irregularity = pd.read_csv(f"input/irregularity_{track}.csv")
df_irregularity_phase_modified = pd.read_csv(f"input/irregularity_{track}_phase_modified.csv")

df_irregularity_phase_modified.head()
df_irregularity_phase_modified.tail()
df_irregularity_phase_modified.shape








## 初期処理 ===================================================================================
## 設定ファイル読み込み
config = configparser.ConfigParser()
config.read(f"scripts/conf_test/track_{track}_test.ini", 'UTF-8')

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
n_diff = config.getint('model', 'n_diff')

# [post]
tol_abnormal_max_min = config.getfloat('post', 'tol_abnormal_max_min')
tol_abnormal_upper = config.getint('post', 'tol_abnormal_upper')
tol_abnormal_lower = config.getint('post', 'tol_abnormal_lower')
method_post = config.get('post', 'method_post')

# [others]
lag_t = config.getint('others', 'lag_t')
org_start_date_id = config.getint('others', 'start_date_id')




## 予測対象キロ程，予測期間の設定
#target_milage_id_list = range(6935,6945)
#target_milage_id_list = range(6900,6970)
#target_milage_id_list = range(6800,7200)
#target_milage_id_list = range(6970,7100)
#target_milage_id_list = range(1700,1730)#1723
#target_milage_id_list = range(8500,8560)#1723
target_milage_id_list = range(80,180)
#target_milage_id_list = range(df_irregularity_phase_modified.shape[1])

t_pred = 91#41
start_date_id = org_start_date_id -1 - lag_t # start_date_id日目の原系列，差分系列を初期値とする＝＞start_date_id+1日目から予測
test_date_id_list = range(start_date_id+1+lag_t, start_date_id+1+lag_t+t_pred)




## テスト用のスモールデータ取得=========================================
df_irregularity_small = deepcopy(df_irregularity.iloc[:,target_milage_id_list])
df_irregularity_phase_modified_small = deepcopy(df_irregularity_phase_modified.iloc[:,target_milage_id_list])



## 前処理 ==================================================================================
print("\n・前処理 ===============================")
time.sleep(0.5)

# 0時点の原系列，差分系列をまとめる
org_dict = {}
org_dict["raw0"] = deepcopy(df_irregularity_phase_modified_small)

# 原系列の前処理：移動平均
org_dict["raw0_prior_treated"], tmp_raw0_median, tmp_raw0_median_diff  = prior_funcs.priorRawData(df_raw=org_dict["raw0"], window=window, min_periods=min_periods, center=center, tol_diff=0.7, tol_n_group=5)


# 原系列前処理結果の保存
folder_name = "output_test/movie_prior"
prior_funcs.makeNewFolder(folder_name)
org_dict["raw0"].to_csv(f"{folder_name}/raw0.csv",index=False,header=True)
tmp_raw0_median.to_csv(f"{folder_name}/tmp_raw0_median.csv",index=False,header=True)
tmp_raw0_median_diff.to_csv(f"{folder_name}/diff_prior_treated.csv",index=False,header=True)
org_dict["raw0_prior_treated"].to_csv(f"{folder_name}/raw0_prior_treated.csv",index=False,header=True)


#
#milage = "m10141"
#plt.plot(org_dict["raw0"][milage])
#plt.plot(tmp_raw0_median[milage])
##plt.plot(tmp_raw0_median_diff[milage])
##plt.plot(tmp_raw0_median_diff_over_tol_diff)
#plt.plot(org_dict["raw0_prior_treated"][milage])
#


# 差分系列の前処理：絶対値がmu+sigma*tol_sigma超過のデータをNaNに変更
org_dict["diff0"] = prior_funcs.priorDiffData(org_df_raw=org_dict["raw0"],df_raw=deepcopy(org_dict["raw0_prior_treated"]), n_diff=n_diff, tol_sigma=tol_sigma_diff_prior, window=window_diff, min_periods=min_periods_diff, center=center_diff)    

# n_diff+1期分の差分系列をまとめる
for i in range(n_diff):
    org_dict[f"diff{i+1}"] = org_dict["diff0"].shift(i+1)



## 初期値の準備
print("\n原系列初期値を取得")
time.sleep(0.5)
start_raw_dict = prior_funcs.makeStartRawDict(df_raw=df_irregularity_small, start_date_id=start_date_id, start_period=start_period, n_average_date=n_average_date, start_average_method=start_average_method)


print("\n差分系列初期値を取得")
time.sleep(0.5)
start_diff_dict, start_values_result_dict = prior_funcs.makeStartDiffDict(df_dict=org_dict, n_diff=n_diff, start_date_id=start_date_id)



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
print("ARIMA")
time.sleep(0.5)
df_pred_raw_lm   = model_funcs.predWithARIMA(train_dict=train_dict, start_raw_dict=start_raw_dict, start_diff_dict=start_diff_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred+lag_t, model_name="lm", n_org_train_dict=n_org_train_dict)

print("直近5日間中央値")
time.sleep(0.5)
df_pred_raw_mean = model_funcs.predWithARIMA(train_dict=train_dict, start_raw_dict=start_raw_dict, start_diff_dict=start_diff_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred+lag_t, model_name="median", n_org_train_dict=n_org_train_dict)



## 後処理 ==================================================================================
print("\n・後処理 ===============================")
time.sleep(0.5)

# 予測結果を検査し，異常を取得
abnormal_total, diagnosis_result = post_funcs.diagnosePredResult(df_pred=deepcopy(df_pred_raw_lm), df_train=deepcopy(train_dict["raw0_prior_treated"]), tol_abnormal_max_min = tol_abnormal_max_min, tol_abnormal_upper = tol_abnormal_upper, tol_abnormal_lower = tol_abnormal_lower)

# 異常値の除去
df_pred_raw_lm = post_funcs.postTreat(df_pred_raw=df_pred_raw_lm, abnormal_total=abnormal_total, start_raw_dict=start_raw_dict, t_pred=t_pred+lag_t, method=method_post)

# 予測対象範囲外の除去
df_pred_raw_lm = df_pred_raw_lm.iloc[range(lag_t, t_pred+lag_t),:]
df_pred_raw_mean = df_pred_raw_mean.iloc[range(lag_t, t_pred+lag_t),:]



## 結果 =============================================================
# 出力ファイル作成
folder_name = "output_test/pred_result"
prior_funcs.makeNewFolder(folder_name)

# 評価データを取得 
test_dict = {}
for key in list(org_dict.keys()):
    test_dict[key] = deepcopy(org_dict[key].iloc[test_date_id_list,:])


# MAE計算・プロット
print("MAE lm")
mae_dict_lm = {}
for milage in list(df_pred_raw_lm.columns):
    mae_dict_lm[milage] = test_funcs.calcMAE(df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_lm[milage])
test_funcs.plotTotalMAE(mae_dict=mae_dict_lm, ylim=[0.0, 1.0], r_plot_size=1, output_dir=f"{folder_name}/ARIMA")

print("MAE mean")
mae_dict_mean = {}
for milage in list(df_pred_raw_mean.columns):
    mae_dict_mean[milage] = test_funcs.calcMAE(df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_mean[milage])
test_funcs.plotTotalMAE(mae_dict=mae_dict_mean, ylim=[0.0, 1.0], r_plot_size=1, output_dir=f"{folder_name}/median")


# lmとmeanのMAEの差分を計算・プロット(-の値の部分でlmが優っている)
diff_mae = np.array(list(mae_dict_lm.values())) - np.array(list(mae_dict_mean.values()))
plt.plot(diff_mae, color="red");plt.ylim([-0.1, 0.1]);plt.grid();plt.show()


#
### 動画用データ保存 =================================================== 
#folder_name = "output_test/movie_pred_result"
#prior_funcs.makeNewFolder(folder_name)
#shutil.copyfile(f"input/irregularity_{track}.csv", f"{folder_name}/org_raw0.csv")
#org_dict["raw0"].to_csv(f"{folder_name}/raw0.csv",index=False,header=True)
#org_dict["raw0_prior_treated"].to_csv(f"{folder_name}/raw0_prior_treated.csv",index=False,header=True)
#train_dict["raw0"].to_csv(f"{folder_name}/train.csv",index=True,header=True)
#train_dict["raw0_prior_treated"].to_csv(f"{folder_name}/train_prior_treated.csv",index=True,header=True)
#df_pred_raw_lm.to_csv(f"{folder_name}/pred_ARIMA.csv",index=True,header=True)
#





### 動画用データ保存 =================================================== 
#folder_name = "output_test/movie_pred_result5"
#prior_funcs.makeNewFolder(folder_name)
#
##diff_mae_1 = deepcopy(diff_mae)
##tmp_nan = np.isnan(diff_mae_1)
##diff_mae_1[tmp_nan] = 0
##
##tmp_bool1 = diff_mae_1 > 0.1
#
#shutil.copyfile(f"input/irregularity_{track}.csv", f"{folder_name}/org_raw0.csv")
#org_dict["raw0"].loc[:,tmp_bool1].to_csv(f"{folder_name}/raw0.csv",index=False,header=True)
#org_dict["raw0_prior_treated"].loc[:,tmp_bool1].to_csv(f"{folder_name}/raw0_prior_treated.csv",index=False,header=True)
#train_dict["raw0"].loc[:,tmp_bool1].to_csv(f"{folder_name}/train.csv",index=True,header=True)
#train_dict["raw0_prior_treated"].loc[:,tmp_bool1].to_csv(f"{folder_name}/train_prior_treated.csv",index=True,header=True)
#df_pred_raw_lm.loc[:,tmp_bool1].to_csv(f"{folder_name}/pred_ARIMA.csv",index=True,header=True)



### 結果をプロット ======================================================================
#milage_id_list = range(17,20)
#ylim=[-10,10]
#for milage_id in milage_id_list:
#    milage = list(df_pred_raw_lm.columns)[milage_id]
#    print(f"\n{milage_id} {milage} ======================================================\n")
#    print("lm")
#        
#    test_funcs.PlotTruthPred(df_train=train_dict["raw0"][milage], df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_lm[milage], inspects_dict=None, 
#            ylim=ylim, r_plot_size=1,output_dir=None, file_name=f"{milage_id}_{milage}")
#    
#    print("mean")
#    test_funcs.PlotTruthPred(df_train=train_dict["raw0"][milage], df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_mean[milage], inspects_dict=None, 
#            ylim=ylim, r_plot_size=1,output_dir=None, file_name=f"{milage_id}_{milage}")
        
    







## ニューラルネットワーク ========================================

from scripts.func import model_funcs


# milage_list_listの作成
n_state = 1
stride = n_state*2 + 1 # 重複無し
#stride = 1             # 重複あり
tol_n_raw = 30
batch_size = 10
epochs = 100



model_spatialAriNnet = nnet_model.spatialAriNnet(input_shape=(n_state*2+1,3))
model_spatialAriNnet.summary()


## 予測モデル作成・逐次予測===============================================================
milage_list_list = model_funcs.makeMilageListList(n_state=n_state, stride=stride, tol_n_raw=tol_n_raw, n_org_train_dict=n_org_train_dict)
df_pred_raw_NN = model_funcs.predWithSpatialAriNnet(milage_list_list=milage_list_list, train_dict=train_dict, start_raw_dict=start_raw_dict, start_diff_dict=start_diff_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred+lag_t, batch_size=batch_size, epochs=epochs)



## 後処理 ==================================================================================
# 予測結果を検査し，異常を取得
abnormal_total, diagnosis_result = post_funcs.diagnosePredResult(df_pred=deepcopy(df_pred_raw_NN), df_train=deepcopy(train_dict["raw0_prior_treated"]), tol_abnormal_max_min = tol_abnormal_max_min, tol_abnormal_upper = tol_abnormal_upper, tol_abnormal_lower = tol_abnormal_lower)

# 異常値の除去
df_pred_raw_NN = post_funcs.postTreat(df_pred_raw=df_pred_raw_NN, abnormal_total=abnormal_total, start_raw_dict=start_raw_dict, t_pred=t_pred+lag_t, method=method_post)

# 予測対象範囲外の除去
df_pred_raw_NN = df_pred_raw_NN.iloc[range(lag_t, t_pred+lag_t),:]



## 動画用データ保存 =================================================== 
folder_name = "output_test/movie_pred_result_NN"
prior_funcs.makeNewFolder(folder_name)

#diff_mae_1 = deepcopy(diff_mae)
#tmp_nan = np.isnan(diff_mae_1)
#diff_mae_1[tmp_nan] = 0
#
#tmp_bool1 = diff_mae_1 > 0.1

shutil.copyfile(f"input/irregularity_{track}.csv", f"{folder_name}/org_raw0.csv")
org_dict["raw0"].to_csv(f"{folder_name}/raw0.csv",index=False,header=True)
org_dict["raw0_prior_treated"].to_csv(f"{folder_name}/raw0_prior_treated.csv",index=False,header=True)
train_dict["raw0"].to_csv(f"{folder_name}/train.csv",index=True,header=True)
train_dict["raw0_prior_treated"].to_csv(f"{folder_name}/train_prior_treated.csv",index=True,header=True)
df_pred_raw_NN.to_csv(f"{folder_name}/pred_NN.csv",index=True,header=True)
df_pred_raw_lm.to_csv(f"{folder_name}/pred_ARIMA.csv",index=True,header=True)




## 結果 =============================================================

# MAE計算・プロット
print("MAE NN")
mae_dict_NN = {}
for milage in list(df_pred_raw_NN.columns):
    mae_dict_NN[milage] = test_funcs.calcMAE(df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_NN[milage])
test_funcs.plotTotalMAE(mae_dict=mae_dict_NN, ylim=[0.0, 1.0], r_plot_size=1, output_dir=f"{folder_name}/ARIMA")


print("MAE lm")
mae_dict_lm = {}
for milage in list(df_pred_raw_lm.columns):
    mae_dict_lm[milage] = test_funcs.calcMAE(df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_lm[milage])
test_funcs.plotTotalMAE(mae_dict=mae_dict_lm, ylim=[0.0, 1.0], r_plot_size=1, output_dir=f"{folder_name}/median")


# lmとmeanのMAEの差分を計算・プロット(-の値の部分でlmが優っている)
diff_mae = np.array(list(mae_dict_NN.values())) - np.array(list(mae_dict_lm.values()))
plt.plot(diff_mae, color="red");plt.ylim([-0.1, 0.1]);plt.grid();plt.show()






print("MAE mean")
mae_dict_mean = {}
for milage in list(df_pred_raw_mean.columns):
    mae_dict_mean[milage] = test_funcs.calcMAE(df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_mean[milage])
test_funcs.plotTotalMAE(mae_dict=mae_dict_mean, ylim=[0.0, 1.0], r_plot_size=1, output_dir=f"{folder_name}/median")


# lmとmeanのMAEの差分を計算・プロット(-の値の部分でlmが優っている)
diff_mae = np.array(list(mae_dict_NN.values())) - np.array(list(mae_dict_mean.values()))
plt.plot(diff_mae, color="red");plt.ylim([-0.1, 0.1]);plt.grid();plt.show()




## 結果をプロット ======================================================================
milage_id_list = range(80,100)
ylim=[-10,10]
for milage_id in milage_id_list:
    milage = list(df_pred_raw_lm.columns)[milage_id]
    print(f"\n{milage_id} {milage} ======================================================\n")
    print("NN")
    test_funcs.PlotTruthPred(df_train=train_dict["raw0"][milage], df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_NN[milage], inspects_dict=None, 
            ylim=ylim, r_plot_size=1,output_dir=None, file_name=f"{milage_id}_{milage}")


    print("lm")
    test_funcs.PlotTruthPred(df_train=train_dict["raw0"][milage], df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_lm[milage], inspects_dict=None, 
            ylim=ylim, r_plot_size=1,output_dir=None, file_name=f"{milage_id}_{milage}")
    
    print("mean")
    test_funcs.PlotTruthPred(df_train=train_dict["raw0"][milage], df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_mean[milage], inspects_dict=None, 
            ylim=ylim, r_plot_size=1,output_dir=None, file_name=f"{milage_id}_{milage}")
#        