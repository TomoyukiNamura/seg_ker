#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 17:06:23 2019

@author: tomoyuki
"""


import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input,Dense,Reshape

# 全結合ニューラルネット
input_shape = (784,)
inputs = Input(shape=input_shape)

x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)

predictions = Dense(10, activation='softmax')(x)

model = Model(input=inputs, output=predictions)
model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

model.summary()


