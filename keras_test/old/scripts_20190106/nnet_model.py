#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras import optimizers
from keras import losses
from keras.models import Model
from keras.layers import Input,Dense,Lambda,Flatten
from keras.layers.local import LocallyConnected1D
from keras.backend import temporal_padding


def spatialAriNnet(input_shape):
    inputs = Input(shape=input_shape)

    # パディング+局所結合層
    x = Lambda(lambda x: temporal_padding(x, padding=(1, 1)))(inputs)
    x = LocallyConnected1D(1, 3, padding='valid', activation='linear')(x)
    
#    # パディング+局所結合層（出力）
#    x = Lambda(lambda x: temporal_padding(x, padding=(1, 1)))(x)
#    predictions = LocallyConnected1D(1, 3, padding='valid', activation='linear')(x)
    
    predictions = Flatten()(x)
    
#    # 全結合層（出力）
#    x = Flatten()(x)
#    predictions = Dense(input_shape[0], activation='linear')(x)
    
    model = Model(input=inputs, output=predictions)
    
    
    ## 最適化手法
    opt = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    # opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # opt = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    # opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    # opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # opt = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    # opt = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    
    
    ## 損失関数
    #loss_func = losses.mean_squared_error
    loss_func = losses.mean_absolute_error
    #loss_func = losses.mean_absolute_percentage_error
    #loss_func = losses.mean_squared_logarithmic_error
    #loss_func = losses.squared_hinge
    #loss_func = losses.hinge
    #loss_func = losses.categorical_hinge
    #loss_func = losses.logcosh
    
    # コンパイル
    model.compile(optimizer=opt, loss=loss_func, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
    
    return model

