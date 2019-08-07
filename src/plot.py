#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

APP_NAME = "/Users/tomoyuki/python_workspace/takeda"
os.chdir(APP_NAME)

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

# 連続・離散データを識別（説明変数の型で決定）
is_continuous = df_data.dtypes=="float64"

# 説明変数を連続データのみにする
df_data = df_data.loc[:, is_continuous]

# 四分位範囲(75%点-25%点)を計算
UPPER_Q = .75
LOWER_Q = .25
quantile_ranges = df_data.quantile(UPPER_Q) - df_data.quantile(LOWER_Q)

# 四分位範囲が0より大きいかを識別
is_large_quantile_ranges = quantile_ranges > 0
df_data = df_data.loc[:, is_large_quantile_ranges]


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