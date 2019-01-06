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


n_diff = 3
df_spatio_diff = {}

for i in range(n_diff+1):
    df_spatio_diff[f"diff{i}"] = df_spatio.shift(i).drop(index=range(n_diff))




import os
os.chdir("/Users/tomoyuki/Desktop/keras_test")
import scripts.model as model

# dfから入力データ作成
y, X = model.dfDict2SAMInput(df_diff=df_spatio_diff)

# spatialARIモデル作成
spatialARIModel = model.spatialARIModel(input_shape=(X.shape[1],X.shape[2]))
spatialARIModel.summary()

# 学習
model.fit(x=X, y=y, batch_size=10, epochs=10, verbose=1)
model.get_weights()



