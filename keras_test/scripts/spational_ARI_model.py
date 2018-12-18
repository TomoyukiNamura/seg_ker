#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 20:31:57 2018

@author: tomoyuki
"""

import numpy as np
from numpy.random import randn,rand
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# 0時点変化量と1時点変化量のモデル
def model(t0,a,b):
    t1 = (a + b * t0 + randn(1)*0.2)[0]
    return t1

# 初期変化量をinit_diffとして，n時点までの変化量をシミュレート
a = -0.00187163
b = -0.42820586
init_diff = a*(1/(1-b))
n = 1000

print(init_diff)
diff_list = []
diff_list.append(init_diff)
for i in range(n):
    diff_list.append(model(diff_list[i],a,b))

df_diff = pd.DataFrame({"diff":diff_list})

# シミュレートしたn時点までの変化量をプロット
plt.plot(df_diff);plt.grid();plt.show();

# 0時点変化量と1時点変化量をプロット
plt.scatter(df_diff.shift(),df_diff);plt.grid();plt.show();


# 0時点・1時点変化量と2時点変化量をプロット
df = pd.DataFrame({"t2":df_diff.iloc[:,0],
                   "t1":np.round(df_diff.shift(1).iloc[:,0],1),
                   "t0":np.round(df_diff.shift(2).iloc[:,0],1)})
    
df_pivot = pd.pivot_table(data=df , values='t2', columns='t0', index='t1', aggfunc=np.mean)

sns.heatmap(df_pivot, annot=False, fmt="1.1f", linewidths=.5, cmap="YlOrRd_r", vmin=min(df_pivot),vmax=max(df_pivot))
plt.show()


# 初期値をinit_orgとして原系列を計算
init_org = 2

org_list = []
org_list.append(init_org)

for i in range(n):
    org_list.append(org_list[i]+diff_list[i])
    
df_org = pd.DataFrame({"org":org_list})
plt.plot(df_org);plt.grid();plt.ylim([-5,5]);plt.show();



df_spatio = []
for i in range(10):
    df_spatio.append(df_diff)
df_spatio = pd.concat(df_spatio,axis=1)


n_diff = 2
df_spatio_diff = {}

for i in range(n_diff+1):
    df_spatio_diff[f"diff{i}"] = df_spatio.shift(i).drop(index=range(n_diff))



## インプットデータ
y = np.array(df_spatio_diff["diff0"])
y = np.reshape(y,(y.shape[0],y.shape[1],1))

X = []
for i in range(1,n_diff+1):
    X.append(np.array(df_spatio_diff[f"diff{i}"]))
X = np.dstack(X)

print(y.shape)
print(X.shape)


from keras import optimizers
from keras.models import Model,Sequential
from keras.layers import Input,Dense,Lambda,Conv1D,Conv2D
from keras.layers.local import LocallyConnected1D, LocallyConnected2D
from keras.layers.core import Activation
from keras.backend import temporal_padding
from keras.wrappers.scikit_learn import KerasRegressor


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


model = spatialARIModel(input_shape=(X.shape[1],X.shape[2]))
model.summary()



model.fit(x=X, y=y, batch_size=10, epochs=10, verbose=1)
model.get_weights()




## 以下参考 =====================================================================================================================
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
    
    # Functional API Model 
    inputs = Input(shape=input_shape)
    x = Dense(100, activation='linear')(inputs)
    predictions = Dense(1, activation='linear')(x)
    model = Model(input=inputs, output=predictions)
    
#    # Sequential Model
#    model = Sequential()
#    model.add(Dense(100, input_shape=input_shape, activation='linear'))
#    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer="rmsprop")
    
    return model



model = reg_model(input_shape=(1,))
model.summary()

model.fit(x=train_x, y=train_y, batch_size=10, epochs=10, verbose=1)
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
    
