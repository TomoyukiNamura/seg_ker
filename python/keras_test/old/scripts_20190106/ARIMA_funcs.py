#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression as lm
from sklearn.linear_model import Ridge,Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.metrics import mean_absolute_error

import scripts.nnet_model as nnet_model



"""
 
 ARIMAモデル用関数 
 
"""


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

    
# 作成済みモデルと初期値を用いて，逐次予測を実行
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

# 初期値のみで逐次予測を実行
def predOnlyStart(start_raw, t_pred):
    pred_raw_list = []
    for i in range(t_pred):
        pred_raw_list.append(start_raw)
    return pred_raw_list
        

def predWithARIMA(train_dict, start_raw_dict, start_diff_dict, n_diff, start_date_id, t_pred, model_name, n_org_train_dict):
        
    df_pred_raw = []
    
    for milage in tqdm(list(train_dict["raw0"].columns)):
        
        # 初期値を取得
        start_raw = start_raw_dict[milage]
        start_diff = start_diff_dict[milage]
        
        # 訓練データ作成
        y, X = dfDict2ARIMAInput(df_dict=train_dict, n_diff=n_diff, milage=milage)
        
        # モデル学習・逐次予測(訓練データ数が30以上，かつ訓練期間の原系列数が30以上のものに限りモデル作成)
        if X.shape[0] >= 30 and n_org_train_dict[milage] >= 30:
            if model_name=="lm":
                # 学習
                model = lm()
                model.fit(X=X, y=y)
                
                # 逐次予測
                pred_raw_list = recursivePredARIMA(model=model, start_diff=start_diff, start_raw=np.reshape(np.array(start_raw), (1,1)), t_pred=t_pred)
            
            elif model_name=="Ridge":
                # 学習
                model = Ridge(alpha=1.0)
                model.fit(X=X, y=y)
                
                # 逐次予測
                pred_raw_list = recursivePredARIMA(model=model, start_diff=start_diff, start_raw=np.reshape(np.array(start_raw), (1,1)), t_pred=t_pred)
                
            elif model_name=="SVR":
                # 学習
                model = SVR(kernel="linear", C=10, epsilon=0.8)
                model.fit(X=X, y=y)
                
                # 逐次予測
                pred_raw_list = recursivePredARIMA(model=model, start_diff=start_diff, start_raw=np.reshape(np.array(start_raw), (1,1)), t_pred=t_pred)
            
            else:
                # 初期値のみで予測
                pred_raw_list = predOnlyStart(start_raw, t_pred)
                    
        else:
            # 初期値のみで予測
            #print("訓練データ数が30未満，または訓練期間の原系列数が30未満")
            pred_raw_list = predOnlyStart(start_raw, t_pred)
            
        
        df_pred_raw.append(pd.DataFrame({milage:pred_raw_list}))
        
    df_pred_raw = pd.concat(df_pred_raw, axis=1)
    df_pred_raw.index = pd.RangeIndex(start=start_date_id+1, stop=start_date_id+1+t_pred, step=1)
    
    return df_pred_raw





"""

 
 nnetモデル用関数 
 

"""


## milage_list_listの作成
def makeMilageListList(n_state, stride, tol_n_raw, n_org_train_dict):
    
    def getForwardMilageList(org_milage_list, milage_id, n_state):
        forward_milage_list = []
        for i in reversed(list(range(n_state))):
            if 0 <= milage_id-(i+1):
                forward_milage_list.append(org_milage_list[milage_id-(i+1)])
        return forward_milage_list
    
    
    def getBackMilageList(org_milage_list, milage_id, n_state):
        back_milage_list = []
        for i in list(range(n_state)):
            if len(org_milage_list) > milage_id+(i+1):
                back_milage_list.append(org_milage_list[milage_id+(i+1)])
        return back_milage_list
    
    org_milage_list =  list(n_org_train_dict.keys())
    
    milage_id_list = list(range(0,len(org_milage_list),stride))
    if (len(org_milage_list)-1 in milage_id_list)==False:
        milage_id_list.append(len(org_milage_list)-1)
    
    milage_list_list = []
    
    for milage_id in milage_id_list:    
        # milage_listを作成
        milage_list = []
        milage_list.extend(getForwardMilageList(org_milage_list, milage_id, n_state))
        milage_list.extend([org_milage_list[milage_id]])
        milage_list.extend(getBackMilageList(org_milage_list, milage_id, n_state))
        
        # milage_list内の全ての原系列データ数がtol_n_raw以上の場合，milage_listにTrueをつける
        tmp_bool = True
        for tmp_milage in milage_list:
            if n_org_train_dict[tmp_milage] < tol_n_raw:
                tmp_bool = False
                break
        
        # 結果を保存
        milage_list_list.append([tmp_bool, milage_list])
    
    return milage_list_list


def dfDict2SpatialAriNnetInput(df_dict, n_diff, milage_list):
    
    # nanを除去
    df_isna = []
    for i in range(n_diff+1):
        df_isna.append(df_dict[f"diff{i}"].loc[:,milage_list].isna().any(axis=1))
    df_isna = pd.concat(df_isna,axis=1)
    df_isna = df_isna.any(axis=1)
    
    # 目的変数作成
    y = np.array(df_dict["diff0"].loc[df_isna==False,milage_list])
    #y = np.reshape(y,(y.shape[0],y.shape[1],1))
    y = np.reshape(y,(y.shape[0],y.shape[1]))
    
    # 説明変数作成
    X = []
    for i in range(1,n_diff+1):
        X.append(np.array(df_dict[f"diff{i}"].loc[df_isna==False,milage_list]))
    X = np.dstack(X)
    
    return y ,X, X.shape[0]

# 対象キロ程(milage_list)のNN逐次予測結果を出力
def recursivePredSpatialAriNnet(milage_list, train_dict, start_raw_dict, start_diff_dict, n_diff, start_date_id, t_pred, batch_size, epochs): 
    
    # 初期値を取得
    start_raw = []
    start_diff = []
    
    for milage in milage_list:
        start_raw.append(start_raw_dict[milage])
        start_diff.append(start_diff_dict[milage])
    
    start_raw = np.array(start_raw)
    start_diff = np.concatenate(start_diff, axis=0)
    start_diff = np.reshape(start_diff, (1,start_diff.shape[0],start_diff.shape[1]))
    
    
    # dfから入力データ作成
    y, X, n_X = dfDict2SpatialAriNnetInput(df_dict=train_dict, n_diff=n_diff, milage_list=milage_list)
    
    # spatialARIモデル作成
    model_spatialAriNnet = nnet_model.spatialAriNnet(input_shape=(X.shape[1],X.shape[2]))
    #model_spatialAriNnet.summary()
    
    # 学習
    model_spatialAriNnet.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.0)
    
    
    # 原系列，差分系列データを初期化
    current_diff = deepcopy(start_diff)
    current_raw  = deepcopy(start_raw)
    
    pred_raw_list = []
    
    for t in range(t_pred):
        # 一期後の差分データを予測
        pred_diff = model_spatialAriNnet.predict(current_diff)
        pred_diff = np.reshape(pred_diff, (pred_diff.shape[1],))
        
        # 一期後の差分により一期後の原系列データを予測
        pred_raw = current_raw + pred_diff
        pred_raw_list.append(np.reshape(pred_raw, (1,pred_raw.shape[0])))
        
        # 差分系列データを更新
        for i in reversed(range(1,current_diff.shape[2])):
            current_diff[0,:,i] = current_diff[0,:,i-1]
        current_diff[0,:,0]  = pred_diff
        
        # 原系列データを更新
        current_raw = pred_raw
            
        
    pred_raw_list = np.concatenate(pred_raw_list, axis=0)
    df_pred_raw_NN = pd.DataFrame(pred_raw_list)
    df_pred_raw_NN.columns = milage_list
    df_pred_raw_NN.index = pd.RangeIndex(start=start_date_id+1, stop=start_date_id+1+t_pred, step=1)
    
    return df_pred_raw_NN

def predWithSpatialAriNnet(milage_list_list, train_dict, start_raw_dict, start_diff_dict, n_diff, start_date_id, t_pred, batch_size, epochs):
        
    org_milage_list = list(train_dict["diff0"].columns)
    
    tmp_df_pred_raw = []
    
    for milage_list in tqdm(milage_list_list):
        if milage_list[0] == True:
            # NNによる逐次予測
            tmp_pred = recursivePredSpatialAriNnet(milage_list=milage_list[1], train_dict=train_dict, start_raw_dict=start_raw_dict, start_diff_dict=start_diff_dict, n_diff=n_diff, start_date_id=start_date_id, t_pred=t_pred, batch_size=batch_size, epochs=epochs)
        
        else:
            tmp_pred = []
            for milage in milage_list[1]:
                # 初期値のみによる予測
                start_raw = start_raw_dict[milage]
                tmp_pred.append(pd.DataFrame({milage:predOnlyStart(start_raw, t_pred)}))
    
            tmp_pred = pd.concat(tmp_pred, axis=1)
            tmp_pred.index = pd.RangeIndex(start=start_date_id+1, stop=start_date_id+1+t_pred, step=1)
      
        tmp_df_pred_raw.append(tmp_pred)
    
    tmp_df_pred_raw = pd.concat(tmp_df_pred_raw, axis=1)
    
    # 結果をまとめる（重複しているキロ程は平均値をとる）
    df_pred_raw = []
    for milage in org_milage_list:
        df_pred_raw.append(tmp_df_pred_raw.loc[:,[milage]].mean(axis=1))
    df_pred_raw = pd.concat(df_pred_raw, axis=1)
    df_pred_raw.columns = org_milage_list
    
    return df_pred_raw




"""

 
 後処理用関数 
 

"""


def diagnosePredResult(df_pred, df_train, tol_abnormal_max_min = 2.5, tol_abnormal_upper = 25, tol_abnormal_lower = -25):
    # 検査結果保存場所
    diagnosis_result = {}
    
    # 最大値-最小値が訓練データの最大値-最小値のtol_abnormal_max_min倍以上のもの
    train_max_min = df_train.max() - df_train.min()
    pred_max_min = df_pred.max() - df_pred.min()
    abnormal_max_min  = pred_max_min > train_max_min*tol_abnormal_max_min
    diagnosis_result["abnormal_max_min"] = abnormal_max_min
    
    # 最大値がtol_abnormal_upperを越すもの
    abnormal_upper = df_pred.max() > tol_abnormal_upper
    diagnosis_result["abnormal_upper"] = abnormal_upper
    
    # 最小値がtol_abnormal_lowerを下回るもの
    abnormal_lower = df_pred.min() < tol_abnormal_lower
    diagnosis_result["abnormal_lower"] = abnormal_lower
    
    # 全ての検査項目の和集合を出力
    abnormal_total = abnormal_max_min | abnormal_upper | abnormal_lower
    
    return abnormal_total, diagnosis_result


def postTreat(df_pred_raw, abnormal_total, start_raw_dict, t_pred, method):
    # 後処理前のデータをコピー
    df_pred_raw_post = deepcopy(df_pred_raw)
    
    # キロ程リスト取得
    milage_list = list(df_pred_raw_post.columns)
        
    for milage_id in tqdm(range(len(milage_list))):
        milage = milage_list[milage_id]
        target_start = start_raw_dict[milage]
        
        if abnormal_total[milage]:
            
            if method=="average":
                # 前後共ない場合，初期値のみで予測結果を修正
                modified_pred = predOnlyStart(start_raw_dict[milage], t_pred)
                
            else:
                # となりのキロ程を取得
                if milage_id==0:
                    next_milage_list = [milage_list[milage_id+1]]
                elif milage_id==(len(milage_list)-1):
                    next_milage_list = [milage_list[milage_id-1]]
                else:
                    next_milage_list = [milage_list[milage_id-1], milage_list[milage_id+1]]
                
                # となりのキロ程にover_tol==Falseがあるかチェック
                tmp_not_over_tol = abnormal_total[next_milage_list]==False
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
            
    return df_pred_raw_post




    
"""

 
 testスクリプト用関数 


"""

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
    

