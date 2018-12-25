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


def predWithARIMA(org_dict, train_dict, n_diff, start_date_id, t_pred, model_name):
        
    df_pred_raw = []
    maked_model_dict = {}
    inspects_dict_dict = {}
    
    for milage in tqdm(list(train_dict["raw0"].columns)):
        
        # 直近10日間平均を原系列初期値に設定(直近10日にデータがない場合，過去5日分データを初期値に利用)
        start_vector = org_dict["raw0"].loc[range(start_date_id-10,start_date_id),milage].dropna()
        if start_vector.shape[0]==0:
             start_vector = org_dict["raw0"].loc[range(start_date_id),milage].dropna().tail(5)
        start_mean = np.mean(np.array(start_vector))
        
        # 訓練データ作成
        y, X = dfDict2ARIMAInput(df_dict=train_dict, n_diff=n_diff, milage=milage)
#        X = np.array([0])
        
        # モデル学習・逐次予測
        if X.shape[0] >= 30:
            if model_name=="lm":
                # 初期値データ作成
                start_raw, start_diff, start_values_result = dfDict2ARIMAInit(df_dict=org_dict, n_diff=n_diff, milage=milage, start_date_id=start_date_id)
                #inspects_dict_dict[milage] = start_values_result
                #inspects_dict_dict[milage]["n_train"] = X.shape[0]
                
                model = lm()
                model.fit(X=X, y=y)
                #maked_model_dict[milage] = model
                
                # 逐次予測
                start_mean = np.reshape(np.array(start_mean), (1,1))
                pred_raw_list = recursivePredARIMA(model=model, start_diff=start_diff, start_raw=start_mean, t_pred=t_pred)
    #            pred_raw_list = recursivePredARIMA(model=model_lm, start_diff=start_diff, start_raw=start_raw, t_pred=t_pred)
                
            elif model_name=="SVR":
                # 初期値データ作成
                start_raw, start_diff, start_values_result = dfDict2ARIMAInit(df_dict=org_dict, n_diff=n_diff, milage=milage, start_date_id=start_date_id)
                inspects_dict_dict[milage] = start_values_result
                inspects_dict_dict[milage]["n_train"] = X.shape[0]
                
                model = SVR(kernel="linear", C=10, epsilon=0.8)
                model.fit(X=X, y=y)
                #maked_model_dict[milage] = model
                
                # 逐次予測
#                start_mean = np.array(np.mean(np.array(org_dict["raw0"].loc[range(start_date_id-10,start_date_id),milage].dropna())))
#                start_mean = np.reshape(start_mean, (1,1))
                start_mean = np.reshape(np.array(start_mean), (1,1))
                pred_raw_list = recursivePredARIMA(model=model, start_diff=start_diff, start_raw=start_mean, t_pred=t_pred)
    #            pred_raw_list = recursivePredARIMA(model=model_lm, start_diff=start_diff, start_raw=start_raw, t_pred=t_pred)
            
            else:
                # 直近10日平均値で逐次予測
                pred_raw_list = []
                for i in range(t_pred):
                    pred_raw_list.append(start_mean)
                    
                inspects_dict_dict = None
                    
        else:
            # 直近10日平均値で逐次予測
            pred_raw_list = []
            for i in range(t_pred):
                pred_raw_list.append(start_mean)
                
            inspects_dict_dict = None
            
        
        df_pred_raw.append(pd.DataFrame({milage:pred_raw_list}))
        
    df_pred_raw = pd.concat(df_pred_raw, axis=1)
    df_pred_raw.index = pd.RangeIndex(start=start_date_id+1, stop=start_date_id+1+t_pred, step=1)
    
    return df_pred_raw, maked_model_dict, inspects_dict_dict





# MAE計算
def calcMAE(df_truth, df_pred):
    tmp_bool = df_truth.isna() == False
    mae = mean_absolute_error(y_true=df_truth[tmp_bool], y_pred=df_pred[tmp_bool])
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
    
    

def postTreat(df_pred_raw, posterior_start_date_id_list, model_name_post, model_name_pred, org_dict, train_dict, n_diff, t_pred):
    # 後処理前のデータをコピー
    df_pred_raw_post = deepcopy(df_pred_raw)
    
    # 後処理用データ取得
    posterior_raw_truth_list = []
    posterior_raw_pred_list = []
    
    for posterior_start_date_id in posterior_start_date_id_list:
        # 実測値の取得
        posterior_date_id_list = range(posterior_start_date_id+1, posterior_start_date_id+1+t_pred)
        posterior_raw_truth_list.append(deepcopy(org_dict["raw0_prior_treated"].iloc[posterior_date_id_list,:]))
        
        # 予測値計算
        posterior_raw_pred,_,_ = predWithARIMA(org_dict=org_dict, train_dict=train_dict, n_diff=n_diff, start_date_id=posterior_start_date_id, t_pred=t_pred, model_name=model_name_pred)    
        posterior_raw_pred_list.append(deepcopy(posterior_raw_pred))
    
    
    
    
    # 後処理用モデル（説：経過日数，目：実測値-予測値）
    model_prior_dict = {}
    for milage in list(posterior_raw_pred.columns):
        
        df_X_y = []
        for i in range(len(posterior_raw_truth_list)):
            # 実測値-予測値　取得
            tmp_truth = posterior_raw_truth_list[i].loc[:,[milage]]
            tmp_pred  = posterior_raw_pred_list[i].loc[:,[milage]]
            tmp_diff_truth_pred = tmp_truth - tmp_pred
                
            # 経過日数　取得
            elapsed_days = pd.DataFrame({"elapsed_days":range(tmp_diff_truth_pred.shape[0])})
            elapsed_days.index = tmp_diff_truth_pred.index
            tmp_diff_truth_pred = pd.concat([elapsed_days, tmp_diff_truth_pred], axis=1)
            
            tmp_diff_truth_pred.index = pd.RangeIndex(start=0,stop=tmp_diff_truth_pred.shape[0],step=1)
            df_X_y.append(deepcopy(tmp_diff_truth_pred))
        
        df_X_y = pd.concat(df_X_y, axis=0).dropna()
            
        # 訓練データ作成
        X = np.array(df_X_y.loc[:,["elapsed_days"]])
        y = np.array(df_X_y.loc[:,[milage]])
        
        # 後処理モデル作成
        if model_name_post == "lm":
            model_post = lm()
            
        elif model_name_post == "knn":
            model_post = knn(n_neighbors=10)
            
        else:
            model_post = lm()
        
        model_prior_dict[milage] = model_post.fit(X=X, y=y)
    
        # 予測結果にゲタ履かせ
        X_pred = np.reshape(np.array(range(t_pred)), (t_pred, 1))
        inflate_series = pd.DataFrame(model_post.predict(X_pred))[0]
        inflate_series.index = df_pred_raw_post.loc[:,milage].index
        
        df_pred_raw_post.loc[:,milage] = df_pred_raw_post.loc[:,milage] + inflate_series
        
    return df_pred_raw_post