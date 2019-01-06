#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
 
 モデル作成用関数 
 
"""


import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import LinearRegression as lm
from sklearn.linear_model import Ridge,Lasso
from sklearn.svm import SVR

import nnet_model
#import scripts.nnet_model as nnet_model



## ARIMAモデル用関数 =====================================================================


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







## nnetモデル用関数 =====================================================================

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


