#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
 
 モデル作成用関数 
 
"""


import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import Ridge

#import nnet_model
#import scripts.func.nnet_model as nnet_model



## ARIMAモデル用関数 =====================================================================


### ARIMA(n_diff,1,0)の訓練データ作成
#def dfDict2ARIMAInput(df_dict, n_diff, milage):
#    
#    # nanを除去
#    df_y_X = []
#    for i in range(n_diff+1):
#        df_y_X.append(df_dict[f"diff{i}"].loc[:,[milage]])
#    df_y_X = pd.concat(df_y_X, axis=1).dropna()
#    
#    # 目的変数(0時点差分データ)を作成
#    y = np.array(df_y_X.iloc[:,[0]])
#    
#    # 説明変数(-n_diff ~ -1時点差分データ)を作成
#    X = []
#    for i in range(1,n_diff+1):
#        X.append(np.array(df_y_X.iloc[:,i]))
#    X = np.dstack(X)[0]
#        
#    return y ,X


    
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
        


#def predWithARIMA(train_dict, start_raw_dict, start_diff_dict, n_diff, start_date_id, t_pred, model_name, n_org_train_dict):
#        
#    df_pred_raw = []
#    
#    for milage in tqdm(list(train_dict["raw0"].columns)):
#        
#        # 初期値を取得
#        start_raw = start_raw_dict[milage]
#        start_diff = start_diff_dict[milage]
#        
#        # 訓練データ作成
#        y, X = dfDict2ARIMAInput(df_dict=train_dict, n_diff=n_diff, milage=milage)
#        
#        
#        # モデル学習・逐次予測(訓練データ数が30以上，直近180日間の原系列数が30以上のものに限りモデル作成)
#        if X.shape[0] >= 10 and n_org_train_dict[milage] >= 10:
#            if model_name=="lm":
#                # 学習
#                model = lm()
#                model.fit(X=X, y=y)
#                
#                # 逐次予測
#                pred_raw_list = recursivePredARIMA(model=model, start_diff=start_diff, start_raw=np.reshape(np.array(start_raw), (1,1)), t_pred=t_pred)
#            
#            elif model_name=="Ridge":
#                # 学習
#                model = Ridge(alpha=0.5)
#                model.fit(X=X, y=y)
#                
#                # 逐次予測
#                pred_raw_list = recursivePredARIMA(model=model, start_diff=start_diff, start_raw=np.reshape(np.array(start_raw), (1,1)), t_pred=t_pred)
#                
#            elif model_name=="SVR":
#                # 学習
#                model = SVR(kernel="linear", C=10, epsilon=0.8)
#                model.fit(X=X, y=y)
#                
#                # 逐次予測
#                pred_raw_list = recursivePredARIMA(model=model, start_diff=start_diff, start_raw=np.reshape(np.array(start_raw), (1,1)), t_pred=t_pred)
#            
#            else:
#                # 初期値のみで予測
#                pred_raw_list = predOnlyStart(start_raw, t_pred)
#                    
#        else:
#            # 初期値のみで予測
#            #print("訓練データ数が30未満，または訓練期間の原系列数が30未満")
#            pred_raw_list = predOnlyStart(start_raw, t_pred)
#            
#        
#        df_pred_raw.append(pd.DataFrame({milage:pred_raw_list}))
#        
#    df_pred_raw = pd.concat(df_pred_raw, axis=1)
#    df_pred_raw.index = pd.RangeIndex(start=start_date_id+1, stop=start_date_id+1+t_pred, step=1)
#    
#    return df_pred_raw




## (変更)
def predWithARIMA(model_dict, init_raw_dict, init_diff_dict, t_pred):
        
    df_pred_raw = []
    
    for milage in tqdm(list(model_dict.keys())):
        
        # モデル・初期値を取得
        model = model_dict[milage]
        init_raw = init_raw_dict[milage]
        init_diff = init_diff_dict[milage]
        
        if model!=None:
            # 逐次予測
            pred_raw_list = recursivePredARIMA(model=model, start_diff=init_diff, start_raw=np.reshape(np.array(init_raw), (1,1)), t_pred=t_pred)
            
        else:
            # 初期値のみで予測
            pred_raw_list = predOnlyStart(init_raw, t_pred)
            
        
        df_pred_raw.append(pd.DataFrame({milage:pred_raw_list}))
        
    df_pred_raw = pd.concat(df_pred_raw, axis=1)
    
    return df_pred_raw



## nnetの関数は全部削除