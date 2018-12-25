#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir("/Users/tomoyuki/Desktop/keras_test")
import numpy as np
from numpy.random import randn,rand
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import time
from sklearn.linear_model import LinearRegression as lm
from sklearn.neighbors import KNeighborsRegressor as knn

from scripts import make_data_funcs
from scripts import ARIMA_funcs

import scripts.model as model


#import configparser
#inifile = configparser.ConfigParser()
#inifile.read('scripts/main.ini', 'UTF-8')
#
#print(type(inifile.get('input', 'file_name')))
#print(type(inifile.get('prior', 'window')))



# データ読み込み
file_name = "track_D.csv"
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


#df_milage_list = {}
#for milage in tqdm(list(df_track.loc[:,"キロ程"].unique())):
#    # 対象キロ程のデータを取得
#    tmp_df = df_track[df_track.loc[:,"キロ程"]==milage]
#    
#    # インデックスを初期化
#    df_milage_list[milage] = tmp_df.reset_index(drop=True)
#
#
#columns = "高低左"
#df_irregularity = []
#for milage in tqdm(list(df_milage_list.keys())):
#    tmp_df = df_milage_list[milage].loc[:,[columns]]
#    df_irregularity.append(tmp_df.rename(columns={columns: f"m{milage}"}))
#
#df_irregularity = pd.concat(df_irregularity, axis=1)
#df_irregularity.head
#df_irregularity.shape




## スモールデータ =====================================
# 訓練データ，評価データの設定
target_milage_id_list = range(1300,1500)

t_pred = 41#91

#start_date_id=15 # start_date_id日目の原系列，差分系列を初期値とする＝＞start_date_id+1日目から予測
start_date_id=200#150

train_date_id_list = range(0,200)

#train_date_id_list = list(range(0, 50))
#train_date_id_list.extend(list(range(100,190)))


test_date_id_list = range(start_date_id+1, start_date_id+1+t_pred)


# 前処理(原系列)の設定
tol_sigma_raw_prior = 2.5
window=30
min_periods=3
center=True

# 前処理(差分系列)の設定
tol_sigma_diff_prior = 1.0
window_diff=3
min_periods_diff=1
center_diff=True

# 予測モデルの設定
model_name_pred = "lm"     #"SVR"
n_diff = 3

## 後処理の設定
#posterior_start_date_id_list = range(110, 200, 30)
#model_name_post = "lm"



## スモールデータ取得=========================================
df_irregularity_small = deepcopy(df_irregularity.iloc[:,target_milage_id_list])

#for milage in list(df_irregularity_small.columns):
#    plt.plot(df_irregularity_small[milage]);plt.ylim([-5,5]);plt.grid();plt.show()



##milage='m22730'
##
#for milage in list(df_irregularity_small.columns):
#    plt.plot(df_irregularity_small[milage].diff());plt.ylim([-0.5,0.5]);plt.title(milage);plt.grid();plt.show()
#    cor_list = []
#    for i in range(30):
#        tmp = df_irregularity_small[milage].diff()
#        tmp_isnotmiss = np.logical_or(tmp.isna(), tmp.shift(i).isna())==False
#        cor_list.append(np.corrcoef(tmp[tmp_isnotmiss], tmp.shift(i)[tmp_isnotmiss])[0][1])
#    
#    plt.plot(cor_list);plt.grid();plt.title(milage);plt.ylim([-1,1]);plt.show()
#
#
##plt.hist(df_irregularity_small[milage].diff(),range=[-0.5,0.5],bins=15);plt.show()





## 前処理 ==================================================================================
print("\n・前処理 ===============================\n")
time.sleep(0.5)

# 0時点の原系列，差分系列をまとめる
org_dict = {}
org_dict["raw0"] = deepcopy(df_irregularity_small)

# 原系列の前処理：移動平均
org_dict["raw0_prior_treated"] = make_data_funcs.priorRawData(df_raw=deepcopy(df_irregularity_small), tol_sigma = tol_sigma_raw_prior, 
                                                              window=window, min_periods=min_periods, center=center)



#milage_list = ["m23803", "m23804", "m23805", "m23806"]
#for milage in milage_list:
#    plt.plot(org_dict["raw0"][milage])
#    plt.plot(org_dict["raw0_prior_treated"][milage])
#    plt.grid();plt.title(milage);plt.show()




# 差分系列の前処理：絶対値がmu+sigma*tol_sigma超過のデータをNaNに変更
org_dict["diff0"] = make_data_funcs.priorDiffData(df_raw=deepcopy(org_dict["raw0_prior_treated"]), n_diff=n_diff, tol_sigma=tol_sigma_diff_prior, window=window_diff, min_periods=min_periods_diff, center=center_diff)
    
#for milage in milage_list:
##    plt.plot(org_dict["raw0"].diff()[milage])
#    plt.plot(org_dict["diff0"][milage])
#    plt.grid();plt.title(milage);plt.show()
##    plt.hist(org_dict["diff0"][milage],range=[-0.01,0.01],bins=15);plt.show()


# n_diff+1期分の差分系列をまとめる
for i in range(n_diff):
    org_dict[f"diff{i+1}"] = org_dict["diff0"].shift(i+1)



#vec_1 = org_dict["diff0"]["m23820"]
#vec_2 = org_dict["diff0"]["m23821"]
#tmp1 = vec_1.isna()==False
#tmp2 = vec_2.isna()==False
#tmp = np.logical_and(tmp1, tmp2)
#np.corrcoef(vec_1[tmp],vec_2[tmp])
#plt.scatter(vec_1,vec_2)

## 予測モデル作成・逐次予測===============================================================
print("\n・予測モデル作成・逐次予測 ===============================\n")
time.sleep(0.5)

# 訓練データを取得 
train_dict = {}
for key in list(org_dict.keys()):
    train_dict[key] = deepcopy(org_dict[key].iloc[train_date_id_list,:])
    

# 評価データを取得 
test_dict = {}
for key in list(org_dict.keys()):
    test_dict[key] = deepcopy(org_dict[key].iloc[test_date_id_list,:])



## ARIMA(n_diff,1,0)による逐次予測
df_pred_raw_lm, maked_model_dict, inspects_dict_dict = ARIMA_funcs.predWithARIMA(org_dict=org_dict, train_dict=train_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred, model_name="lm")
df_pred_raw_mean, _, _ = ARIMA_funcs.predWithARIMA(org_dict=org_dict, train_dict=train_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred, model_name="mean")



### 後処理：予測値と実測値の差分と経過日数の回帰モデルを作成し，予測結果にゲタ履かせ=====================
#print("\n・後処理 ===============================\n")
#time.sleep(0.5)
#
#df_pred_raw = ARIMA_funcs.postTreat(df_pred_raw=df_pred_raw, posterior_start_date_id_list=posterior_start_date_id_list, model_name_post=model_name_post, model_name_pred=model_name_pred, org_dict=org_dict, train_dict=train_dict, n_diff=n_diff, t_pred=t_pred)
#
#print("\n完了 ===============================\n")



## 結果 =============================================================

# MAE計算・プロット
print("MAE lm")
mae_dict_lm = {}
for milage in list(df_pred_raw_lm.columns):
    mae_dict_lm[milage] = ARIMA_funcs.calcMAE(df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_lm[milage])
ARIMA_funcs.plotTotalMAE(mae_dict=mae_dict_lm, ylim=[0.0, 1.0], r_plot_size=1, output_dir=f"ARIMA_C_lm")

print("MAE mean")
mae_dict_mean = {}
for milage in list(df_pred_raw_mean.columns):
    mae_dict_mean[milage] = ARIMA_funcs.calcMAE(df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_mean[milage])
ARIMA_funcs.plotTotalMAE(mae_dict=mae_dict_mean, ylim=[0.0, 1.0], r_plot_size=1, output_dir=f"ARIMA_C_mean")


# lmとmeanのMAEの差分を計算・プロット(-の値の部分でlmが優っている)
diff_mae = np.array(list(mae_dict_lm.values())) - np.array(list(mae_dict_mean.values()))
plt.plot(diff_mae, color="red");plt.ylim([-0.1, 0.1]);plt.grid();plt.show()



# 結果をプロット
milage_id_list = range(125,130)
ylim=[-11,11]
for milage_id in milage_id_list:
    milage = list(df_pred_raw_lm.columns)[milage_id]
    print(f"\n{milage_id} {milage} ======================================================\n")
    print("lm")

    if inspects_dict_dict!=None:
        inspects_dict = inspects_dict_dict[milage]
    else:
        inspects_dict = None
        
    ARIMA_funcs.PlotTruthPred(df_train=train_dict["raw0"][milage], df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_lm[milage], inspects_dict=inspects_dict, 
            ylim=ylim, r_plot_size=1,output_dir=f"ARIMA_C_lm", file_name=f"{milage_id}_{milage}")
    
    print("mean")
    ARIMA_funcs.PlotTruthPred(df_train=train_dict["raw0"][milage], df_truth=test_dict["raw0"][milage], df_pred=df_pred_raw_mean[milage], inspects_dict=inspects_dict, 
            ylim=ylim, r_plot_size=1,output_dir=f"ARIMA_C_mean", file_name=f"{milage_id}_{milage}")
        
    




#
#
#
#
#    
#milage = "m10718"
#
## 結果をプロット
#milage_id_list = range(0,40)
#
#for milage_id in milage_id_list:
#    milage = list(df_pred_raw.columns)[milage_id]
#    ARIMA_funcs.PlotTruthPred(df_train=train_dict["raw0"][milage], df_truth=posterior_dict["raw0"][milage], df_pred=posterior_raw_pred[milage], inspects_dict=None, 
#            ylim=[-5,5], r_plot_size=1,output_dir=None, file_name=f"{milage_id}_{milage}")
#
#    
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

