#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

APP_NAME = "/Users/tomoyuki/python_workspace/takeda"
os.chdir(APP_NAME)

import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


# 設定
DATA_NAME   = "train.csv"
MARGIN_RATE = 0.1
S_RATE      = 0.3
ALPHA       = 0.5

## データ読み込み
df_org = pd.read_csv(f"data/{DATA_NAME}")

# 説明変数、目的変数に分割
df_data = df_org.iloc[:, 2:df_org.shape[1]]
df_target = df_org.iloc[:, [1]]
 
# 四分位範囲が0より大きい連続データ
is_non_range_cont  = utils.check_large_q_cont(df_data, upper_q=.75, lower_q=.25, comparison="equal")

# 離散データ
is_discrete = utils.check_discrete(df_data, tol=20)

# その他(一般の連続データ)
is_cont = (is_non_range_cont==False) & (is_discrete==False)


# データ選択
df_data = df_data.loc[:, is_cont]


# アニメーション作成
margin = (float(np.max(df_target)) - float(np.min(df_target))) * MARGIN_RATE
columns = list(df_data.columns)
fig = plt.figure()

def _plot(data):
    plt.cla()
    column = columns[data]
    plt.scatter(df_data.loc[:,column], df_target, s=plt.rcParams['lines.markersize']**2 * S_RATE, alpha=ALPHA)
    plt.grid()
    plt.title(f"{column}")
    plt.ylim([float(np.min(df_target))-margin, float(np.max(df_target))+margin])
    plt.xlabel(f"{column}")

anim = animation.FuncAnimation(fig, _plot, frames=len(columns))
plt.show()