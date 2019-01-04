#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from copy import deepcopy
from keras import optimizers
from keras.models import Model,Sequential
from keras.layers import Input,Dense,Lambda,Conv1D,Conv2D,Reshape,Flatten
from keras.layers.local import LocallyConnected1D, LocallyConnected2D
from keras.layers.core import Activation
from keras.backend import temporal_padding


def spatialAriNnet(input_shape):
    inputs = Input(shape=input_shape)

    # パディング+局所結合層
    x = Lambda(lambda x: temporal_padding(x, padding=(1, 1)))(inputs)
    x = LocallyConnected1D(1, 3, padding='valid', activation='linear')(x)
    
#    # パディング+局所結合層（出力）
#    x = Lambda(lambda x: temporal_padding(x, padding=(1, 1)))(x)
#    predictions = LocallyConnected1D(1, 3, padding='valid', activation='linear')(x)
    
    # 全結合層（出力）
    x = Flatten()(x)
    predictions = Dense(input_shape[0], activation='linear')(x)
    
    model = Model(input=inputs, output=predictions)
    model.compile(loss='mean_squared_error', optimizer="rmsprop")
    
    return model


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
def recursivePredSpatialAriNnet(milage_list, train_dict, start_raw_dict, start_diff_dict, n_diff, start_date_id, t_pred): 
    
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
    model_spatialAriNnet = spatialAriNnet(input_shape=(X.shape[1],X.shape[2]))
    model_spatialAriNnet.summary()
    
    # 学習
    model_spatialAriNnet.fit(x=X, y=y, batch_size=10, epochs=20, verbose=1)
    
    
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