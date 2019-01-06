#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 20:31:57 2018

@author: tomoyuki
"""

import numpy as np
from numpy.random import randn,rand
import matplotlib.pyplot as plt

from keras import optimizers
from keras.models import Model,Sequential
from keras.layers import Input,Dense
from keras.layers.core import Activation
from keras.wrappers.scikit_learn import KerasRegressor


# テストデータ作成
n_train = 100
n_val = 100

w = 2
all_x = rand(n_train+n_val) * w*np.pi - (w/2.0)*np.pi
all_y = np.sin(all_x) + randn(n_train+n_val)*0.2

train_x = all_x[0:n_train]
train_y = all_y[0:n_train]
plt.scatter(train_x,train_y)

val_x = all_x[n_train:(n_train+n_val)]
val_y = all_y[n_train:(n_train+n_val)]
plt.scatter(val_x,val_y)



def reg_model(input_shape):
    model = Sequential()
    model.add(Dense(100, input_shape=input_shape, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))

    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model



model = reg_model(input_shape=(1,))
model.fit(x=train_x, y=train_y, batch_size=10, epochs=1000, verbose=1)


model.summary()
model.get_weights()


pred_y = model.predict(train_x)
plt.scatter(train_x,train_y)
plt.scatter(train_x,pred_y)



pred_y = model.predict(val_x)
plt.scatter(val_x,val_y)
plt.scatter(val_x,pred_y)





#def reg_model(input_shape):
#    input_ = Input(shape=input_shape)
#    x = input_
#    x = Dense(10)(x)
#    x = Dense(1)(x)
#    x = Activation("linear")(x)
#    model = Model(input_, x)
#    model.compile(optimizer='sgd',loss="mean_squared_error",metrics=['mse','mae'])
#    return model


#def reg_model2():
#    model = Sequential()
#    model.add(Dense(7, input_shape=(1,), activation='tanh'))
#    model.add(Dense(1, activation='linear'))
#
#    # compile model
#    model.compile(loss='mean_squared_error', optimizer='sgd')
#    return model
#model = KerasRegressor(build_fn=reg_model2, batch_size=10, epochs=100, verbose=1)
#model.fit(x=train_x, y=train_y)
    




#model.compile(optimizer='sgd',
#              loss="mean_squared_error",#目的関数
#              metrics=['mae',],#訓練やテストの際にモデルを評価するための評価関数のリスト
#              sample_weight_mode=None, #"temporal"時間ごとのサンプルの重み付け（2次元の重み）
#              weighted_metrics=None, #訓練やテストの際にsample_weightまたはclass_weightにより評価と重み付けされるメトリクスのリスト
#              target_tensors=None)


#model.fit(x=train_x, y=train_x,
#          batch_size=10,
#          epochs=10,
#          verbose=2, #ログ出力の設定
#          callbacks=None,# tensorboardの設定読み込み？
#          class_weight=None,#クラス毎の重みを格納(過小評価されたクラスのサンプルに「より注意を向ける」ようにしたい時に便利です)
#          sample_weight=None,#訓練のサンプルに対する重み
#          validation_split=0.0, # 訓練データ中のvalidationデータの割合
#          validation_data=(val_x, val_x), # varidation data( 設定するとvalidation_splitを上書きする)
#          initial_epoch=0,# 訓練開始時のepoch（前の学習から再開する際に便利です）
#          steps_per_epoch=None,#終了した1エポックを宣言して次のエポックを始めるまでのステップ数の合計
#          validation_steps=None,#停止する前にバリデーションするステップの総数(steps_per_epochを指定している場合のみ関係)
#          shuffle=True, #バッチサイズのチャンクの中においてシャッフル(steps_per_epochがNoneに設定されている場合は効果がありません)
#          )

#pred_y = model.predict(x=val_x,batch_size=10,verbose=1,steps=None)


#
#model = Sequential()
#
#model.add(Dense(10, activation="relu", input_shape=(1,)))
##model.add(Dense(20, activation="relu"))
##model.add(Dense(10, activation="relu"))
#model.add(Dense(1 , activation="softmax"))
#
#model.summary()






from keras.utils.generic_utils import slice_arrays


x=train_x
y=train_y
num_train_samples = 100
batch_size = 3

x, y, sample_weights = model._standardize_user_data(
            x, y,
            sample_weight=None,
            class_weight=None,
            batch_size=batch_size)

model._make_train_function()
fit_function = model.train_function
fit_inputs = x + y + sample_weights

index_array = np.arange(num_train_samples)

num_batches = (num_train_samples + batch_size - 1) // batch_size  # round up
batches = [(i * batch_size, min(num_train_samples, (i + 1) * batch_size))
            for i in range(num_batches)]


for batch_index, (batch_start, batch_end) in enumerate(batches):
    batch_ids = index_array[batch_start:batch_end] 
    ins_batch = slice_arrays(fit_inputs, batch_ids)
    
    outs = fit_function(ins_batch)
    print(outs)
    
