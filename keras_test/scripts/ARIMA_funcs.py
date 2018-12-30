#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression as lm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.metrics import mean_absolute_error

## ARIMA(n_diff,1,0)の訓練データ作成
def dfDict2ARIMAInput(df_dict, n_diff, milage):
    
    # nanを除去
    df_y_X = []
    for i in range(n_diff+1):
        df_y_X.append(df_dict[f"diff{i}"].loc[:,[milage]])
    df_y_X = pd.concat(df_y_X, axis=1).dropna()
    
    # 目的変数(0時点差分データ)を作成
    y = np.array(df_y_X.iloc[:,[0]])
    
    # 説明変数(-n_diff ~ -1時点差分データ)を作成
    X = []
    for i in range(1,n_diff+1):
        X.append(np.array(df_y_X.iloc[:,i]))
    X = np.dstack(X)[0]
        
    return y ,X


## ARIMA(n_diff,1,0)の初期値データ作成
def dfDict2ARIMAInit(df_dict, n_diff, milage, start_date_id):
    # 原系列(NaNの場合は1時点前のデータで補完)
    how_long_back = 0
    start_raw = df_dict["raw0"][milage][start_date_id]
    #start_raw = df_dict["raw0_prior_treated"][milage][start_date_id]
    
    while(np.isnan(start_raw)):
        how_long_back += 1
        start_raw = df_dict["raw0"][milage][start_date_id-how_long_back]
        #start_raw = df_dict["raw0_prior_treated"][milage][start_date_id-how_long_back]
        
    start_raw = np.array([[start_raw]])
        

    # 差分系列(NaNの場合は0で補完)
    n_diff_iszero = 0
    start_diff = []
    
    for i in range(n_diff):
        tmp_data = df_dict[f"diff{i}"][milage][start_date_id]
        
        if np.isnan(tmp_data):
            n_diff_iszero += 1
            start_diff.append(np.array(0))
        else:
            start_diff.append(np.array(tmp_data))
            
    start_diff = np.dstack(start_diff)[0]
    
    # 初期値作成の結果をまとめる
    result = {"how_long_back":how_long_back, "n_diff_iszero":n_diff_iszero}
    
    return start_raw, start_diff, result
    

def recursivePredARIMA(model, start_diff, start_raw, t_pred):
    # 原系列，差分系列データを初期化
    current_diff = deepcopy(start_diff)
    current_raw  = deepcopy(start_raw)
    
    pred_raw_list = []
    for t in range(t_pred):
        # 一期後の差分データを予測
        pred_diff = model.predict(current_diff)
        pred_diff = np.reshape(pred_diff, (1,1))
        
        # 一期後の差分により一期後の原系列データを予測
        pred_raw = current_raw + pred_diff
        pred_raw_list.append(pred_raw[0][0])
        
        # 差分系列データを更新
        for i in reversed(range(1,current_diff.shape[1])):
            current_diff[0][i] = current_diff[0][i-1]
        current_diff[0][0]  = pred_diff[0][0]
        
        # 原系列データを更新
        current_raw = pred_raw
        
    return pred_raw_list


def predOnlyStart(start_median, t_pred):
    pred_raw_list = []
    for i in range(t_pred):
        pred_raw_list.append(start_median)
    return pred_raw_list
        

def predWithARIMA(org_dict, train_dict, n_diff, start_date_id, t_pred, model_name, n_org_train_dict):
        
    df_pred_raw = []
    maked_model_dict = {}
    inspects_dict_dict = {}
    start_raw_dict = {}
    
    for milage in tqdm(list(train_dict["raw0"].columns)):
        
        # nan除く過去5日分中央値を初期値とする
        start_vector = org_dict["raw0"].loc[range(start_date_id+1),milage].dropna().tail(5)
        start_median = np.median(np.array(start_vector))
        start_raw_dict[milage] = deepcopy(start_median)
        
        # 訓練データ作成
        y, X = dfDict2ARIMAInput(df_dict=train_dict, n_diff=n_diff, milage=milage)
        
        # モデル学習・逐次予測(訓練データ数が30以上，かつ訓練期間の原系列数が30以上のものに限りモデル作成)
        if X.shape[0] >= 30 and n_org_train_dict[milage] >= 30 :
            if model_name=="lm":
                # 初期値データ作成
                start_raw, start_diff, start_values_result = dfDict2ARIMAInit(df_dict=org_dict, n_diff=n_diff, milage=milage, start_date_id=start_date_id)
                inspects_dict_dict[milage] = start_values_result
                inspects_dict_dict[milage]["n_train"] = X.shape[0]
                
                model = lm()
                model.fit(X=X, y=y)
                
                # 逐次予測
                pred_raw_list = recursivePredARIMA(model=model, start_diff=start_diff, start_raw=np.reshape(np.array(start_median), (1,1)), t_pred=t_pred)
                
            elif model_name=="SVR":
                # 初期値データ作成
                start_raw, start_diff, start_values_result = dfDict2ARIMAInit(df_dict=org_dict, n_diff=n_diff, milage=milage, start_date_id=start_date_id)
                inspects_dict_dict[milage] = start_values_result
                inspects_dict_dict[milage]["n_train"] = X.shape[0]
                
                model = SVR(kernel="linear", C=10, epsilon=0.8)
                model.fit(X=X, y=y)
                
                # 逐次予測
                pred_raw_list = recursivePredARIMA(model=model, start_diff=start_diff, start_raw=np.reshape(np.array(start_median), (1,1)), t_pred=t_pred)
            
            else:
                # 初期値のみで予測
                pred_raw_list = predOnlyStart(start_median, t_pred)
                inspects_dict_dict[milage] = None
                    
        else:
            # 初期値のみで予測
            #print("訓練データ数が30未満，または訓練期間の原系列数が30未満")
            pred_raw_list = predOnlyStart(start_median, t_pred)
            inspects_dict_dict[milage] = None
            
#        ## 予測結果絶対値が30を超過する場合，平均値に置き換え
#        if any(np.abs(pred_raw_list) > 30):
#            # 初期値のみで予測
#            print("予測結果絶対値が30を超過")
#            pred_raw_list = predOnlyStart(start_median, t_pred)
#            inspects_dict_dict[milage] = None
            
        
        df_pred_raw.append(pd.DataFrame({milage:pred_raw_list}))
        
    df_pred_raw = pd.concat(df_pred_raw, axis=1)
    df_pred_raw.index = pd.RangeIndex(start=start_date_id+1, stop=start_date_id+1+t_pred, step=1)
    
    return df_pred_raw, start_raw_dict, maked_model_dict, inspects_dict_dict





# MAE計算
def calcMAE(df_truth, df_pred):
    tmp_bool = df_truth.isna() == False
    if np.any(tmp_bool)==True:
        mae = mean_absolute_error(y_true=df_truth[tmp_bool], y_pred=df_pred[tmp_bool])
    else:
        mae = np.nan
    return mae

def PlotTruthPred(df_train, df_truth, df_pred, inspects_dict=None, ylim=None, r_plot_size=1, output_dir=None, file_name=""):
        
    # MAE計算
    mae = calcMAE(df_truth, df_pred)
    
    # プロット
    plt.rcParams["font.size"] = 10*r_plot_size
    plt.rcParams['figure.figsize'] = [6.0*r_plot_size, 4.0*r_plot_size]
    
    plt.plot(df_train, label="train", color="black")
    plt.plot(df_truth, label="truth", color="blue")
    plt.plot(df_pred, label="pred", color="red")
    
    if ylim!=None:
        plt.ylim(ylim)
        
    if inspects_dict!=None:
        xlabel = ""
        for key in list(inspects_dict.keys()):
            xlabel = xlabel + f"{key}:{inspects_dict[key]}  "
        plt.xlabel(xlabel)
        
    plt.grid()
    plt.title(file_name + f"    mae: {round(mae, 4)}")    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    if output_dir!=None and file_name!="":
        if os.path.exists(output_dir)==False:
            os.mkdir(output_dir)
            
        plt.savefig(f"{output_dir}/{file_name}.jpg", bbox_inches='tight')
    
    plt.show()
    

def plotTotalMAE(mae_dict, ylim=None, r_plot_size=1, output_dir=None):
        
    mae_vector = np.array(list(mae_dict.values()))
    mae_vector = mae_vector[~np.isnan(mae_vector)]
    
    plt.rcParams["font.size"] = 10*r_plot_size
    plt.rcParams['figure.figsize'] = [6.0*r_plot_size, 4.0*r_plot_size]
    
    plt.plot(mae_vector, color="blue")
    plt.grid()
    plt.title(f"total MAE : {np.round(np.mean(mae_vector),4)}")
    if ylim!=None:
        plt.ylim(ylim)
    
    if output_dir!=None:
        if os.path.exists(output_dir)==False:
            os.mkdir(output_dir)
        plt.savefig(f"{output_dir}/total_MAE.jpg", bbox_inches='tight')
        
    plt.show()
    
    

def postTreat(df_pred_raw, start_raw_dict, t_pred, tol=20):
    # 後処理前のデータをコピー
    df_pred_raw_post = deepcopy(df_pred_raw)
    
    # キロ程リスト取得
    milage_list = list(df_pred_raw_post.columns)
    
    # 予測結果絶対値がtol以上のものを抽出
    over_tol = (np.abs(df_pred_raw_post) >= tol).any(axis=0)
        
    for milage_id in tqdm(range(len(milage_list))):
        milage = milage_list[milage_id]
        target_start = start_raw_dict[milage]
        
        if over_tol[milage]:
            # となりのキロ程を取得
            if milage_id==0:
                next_milage_list = [milage_list[milage_id+1]]
            elif milage_id==(len(milage_list)-1):
                next_milage_list = [milage_list[milage_id-1]]
            else:
                next_milage_list = [milage_list[milage_id-1], milage_list[milage_id+1]]
            
            # となりのキロ程にover_tol==Falseがあるかチェック
            tmp_not_over_tol = over_tol[next_milage_list]==False
            donor_milage_list = list(tmp_not_over_tol[tmp_not_over_tol].index)
            
            if len(donor_milage_list)==2:
                # 前後いずれもある場合，前後の予測結果の平均値+(ターゲットの初期値-前後の初期値の平均値)で修正
                front_pred = deepcopy(df_pred_raw_post.loc[:,donor_milage_list[0]])
                front_start = start_raw_dict[donor_milage_list[0]]
                
                back_pred = deepcopy(df_pred_raw_post.loc[:,donor_milage_list[1]])
                back_start = start_raw_dict[donor_milage_list[1]]
                
                donor_pred = (front_pred + back_pred) / 2.0
                donor_start = (front_start + back_start) / 2.0
                
                modified_pred = donor_pred + (target_start - donor_start)
                
                
            elif len(donor_milage_list)==1:
                # 片一方の場合，ドナーの予測結果+(ターゲットの初期値-ドナーの初期値)で修正
                donor_pred = deepcopy(df_pred_raw_post.loc[:,donor_milage_list[0]])
                donor_start = start_raw_dict[donor_milage_list[0]]
                
                modified_pred = donor_pred + (target_start - donor_start)
            
            else:
                # 前後共ない場合，初期値のみで予測結果を修正
                modified_pred = predOnlyStart(start_raw_dict[milage], t_pred)
            
            # 予測結果を修正
            modified_pred = pd.Series(modified_pred)
            modified_pred.index = df_pred_raw_post[milage].index
            df_pred_raw_post[milage] = modified_pred
            
    return df_pred_raw_post, over_tol