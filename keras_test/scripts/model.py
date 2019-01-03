#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
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
        df_isna.append(df_dict[f"diff{i}"].isna().any(axis=1))
    df_isna = pd.concat(df_isna,axis=1)
    df_isna = df_isna.any(axis=1)
    
    # 目的変数作成
    y = np.array(df_dict["diff0"].loc[df_isna==False,:])
    #y = np.reshape(y,(y.shape[0],y.shape[1],1))
    y = np.reshape(y,(y.shape[0],y.shape[1]))
    
    # 説明変数作成
    X = []
    for i in range(1,n_diff+1):
        X.append(np.array(df_dict[f"diff{i}"].loc[df_isna==False,:]))
    X = np.dstack(X)
    
    return y ,X, X.shape[0]