#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import random
%matplotlib inline
random.seed(0)
# 乱数の係数
random_factor = 0.05
# サイクルあたりのステップ数
steps_per_cycle = 80
# 生成するサイクル数
number_of_cycles = 100

df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
df["sin_t"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)+ random.uniform(-1.0, +1.0) * random_factor))
df[["sin_t"]].head(steps_per_cycle * 2).plot()

def _load_data(data, n_prev = 100):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev = 100):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)



length_of_sequences = 3
(X_train, y_train), (X_test, y_test) = train_test_split(df[["sin_t"]], n_prev =length_of_sequences)  






# 時系列データ分析

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift



n = 1000
n_test = 100

length_of_sequences = 1

y = np.log(np.arange(10,n+10)) + np.random.randn(n)*0.0
plt.plot(y)

df = pd.DataFrame({'y':y})
for i in range(length_of_sequences):
    tmp_df = df['y'].shift(i+1)
    tmp_df.name = f"x{i+1}"
    df = pd.concat([df,tmp_df],axis=1)

df=df.dropna(how='any', axis=0).reset_index(drop=True)


X, y = [], []
for i in range(df.shape[0]):
    tmp_list = []
    for j in range(length_of_sequences):
        tmp_list.append([df.iloc[i,j+1]])
    X.append(tmp_list)
    y.append([df.iloc[i,0]])
    
X = np.array(X)
y = np.array(y)

X_train = X[0:(n-n_test-length_of_sequences-1)]
X_test = X[(n-n_test-length_of_sequences-1+1):n]

y_train = y[0:(n-n_test-length_of_sequences-1)]
y_test = y[(n-n_test-length_of_sequences-1+1):n]



## rnn,lstm
from keras.models import Sequential  
from keras.layers import InputLayer
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM,SimpleRNN


def lstm_model(hidden_neurons, batch_input_shape):
    model = Sequential()  
    
    model.add(InputLayer(batch_input_shape=batch_input_shape))
    
#    model.add(LSTM(hidden_neurons, return_sequences=False)) 
    model.add(SimpleRNN(units=hidden_neurons, return_sequences=False, activation='sigmoid'))
    
    model.add(Dense(in_out_neurons))  
    #model.add(Activation("linear"))  
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model


model = lstm_model(hidden_neurons=4, batch_input_shape=(None, length_of_sequences, 1))
model.summary()
model.fit(X_train, y_train, batch_size=10, epochs=300, validation_split=0.05) 


plt.plot(y_train,label="truth")
plt.plot(model.predict(X_train) ,label="pred")
plt.legend()

plt.plot(y_test,label="truth")
plt.plot(model.predict(X_test) ,label="pred")
plt.legend()



plt.scatter(df['x1'],df['y'])


