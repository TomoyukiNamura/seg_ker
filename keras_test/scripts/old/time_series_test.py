#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 22:41:16 2018

@author: tomoyuki
"""

import numpy as np
from numpy.random import randn,rand
import matplotlib.pyplot as plt
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







import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



def linear3DModel(x1, x2, a, b1, b2):
    y = a + b1*x1 + b2*x2
    return y


def plotLinear3DModel(a, b1, b2):

    x1 = []
    x2 = []
    y = []
    for i in range(len(x_org)):
        for j in range(len(x_org)):
            x1.append(x_org[i])
            x2.append(x_org[j])
            y.append(linear3DModel(x_org[i], x_org[j], a, b1, b2))
    
    
    df = pd.DataFrame({"y":y, "x1":np.round(np.array(x1),1), "x2":np.round(np.array(x2),1)})
    
    df_pivot = pd.pivot_table(data=df , values='y', columns='x1', index='x2', aggfunc=np.mean)
    
    #sns.heatmap(df_pivot, annot=False, fmt="1.1f", linewidths=.5, cmap="YlOrRd_r", vmin=-4,vmax=4)
    sns.heatmap(df_pivot, annot=False, fmt="1.1f", linewidths=.5, cmap="gist_yarg_r", vmin=-4,vmax=4)
    plt.show()


x_org = np.arange(-4,4,0.5)

a = 0


b1 = -0.8
b2 = -0.8

plotLinear3DModel(a, b1, b2)



