#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from keras import optimizers
from keras.models import Model,Sequential
from keras.layers import Input,Dense,Lambda,Conv1D,Conv2D
from keras.layers.local import LocallyConnected1D, LocallyConnected2D
from keras.layers.core import Activation
from keras.backend import temporal_padding


def spatialARIModel(input_shape):
    inputs = Input(shape=input_shape)
    #x = Conv1D(1, 3, padding='same', activation='linear')(inputs)
    #predictions = Conv1D(1, 3, padding='same', activation='linear')(x)
    
    x = Lambda(lambda x: temporal_padding(x, padding=(1, 1)))(inputs)
    x = LocallyConnected1D(1, 3, padding='valid', activation='linear')(x)
    x = Lambda(lambda x: temporal_padding(x, padding=(1, 1)))(x)
    predictions = LocallyConnected1D(1, 3, padding='valid', activation='linear')(x)
    
    model = Model(input=inputs, output=predictions)
    model.compile(loss='mean_squared_error', optimizer="rmsprop")
    
    return model


def dfDict2SAMInput(df_diff):
    y = np.array(df_diff["diff0"])
    y = np.reshape(y,(y.shape[0],y.shape[1],1))
    
    X = []
    for i in range(1,len(df_diff)):
        X.append(np.array(df_diff[f"diff{i}"]))
    X = np.dstack(X)
    
    return y ,X