#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

APP_NAME = "/Users/tomoyuki/python_workspace/takeda"
os.chdir(APP_NAME)
sys.path.append(os.path.join(APP_NAME, "src"))

import utils

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import time
import shutil
import configparser

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.svm import SVR


# 設定
DATA_NAME   = "train.csv"
RANDOM_SEED = 0
TEST_SIZE   = 0.5
MARGIN_RATE = 0.01


# データ読み込み
df_org = pd.read_csv(f"data/{DATA_NAME}")
df_org.shape
df_org.columns

# 説明変数、目的変数に分割
X = df_org.iloc[:, 2:df_org.shape[1]].values
y = df_org.iloc[:, [1]].values

# 訓練・評価データ変換
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
X_train.shape
X_test.shape
y_train.shape
y_test.shape


# 標準化
std_scaler = StandardScaler()
#std_scaler = RobustScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std  = std_scaler.transform(X_test)



# 線形回帰
model_lr    = LinearRegression()
model_lr.fit(X_train_std, y_train)
pred_lr_train    = model_lr.predict(X_train_std)
utils.plot_pred(y_train, pred_lr_train, margin_rate=MARGIN_RATE)
pred_lr_test    = model_lr.predict(X_test_std)
utils.plot_pred(y_test, pred_lr_test, margin_rate=MARGIN_RATE)


# リッジ回帰
ALPHA_RIDGE = 1 * 10 ** 3
#ALPHA_RIDGE = 3 * 10 ** 2
model_ridge = Ridge(alpha= ALPHA_RIDGE)
model_ridge.fit(X_train_std, y_train)
pred_ridge_train = model_ridge.predict(X_train_std)
utils.plot_pred(y_train, pred_ridge_train, margin_rate=MARGIN_RATE)
pred_ridge_test = model_ridge.predict(X_test_std)
utils.plot_pred(y_test, pred_ridge_test, margin_rate=MARGIN_RATE)


# lasso回帰
ALPHA_LASSO = 3 * 10 ** (-3)
#ALPHA_LASSO = 3 * 10 ** (-3)
model_lasso = Lasso(alpha=ALPHA_LASSO)
model_lasso.fit(X_train_std, y_train)
pred_lasso_train = model_lasso.predict(X_train_std)
utils.plot_pred(y_train, pred_lasso_train, margin_rate=MARGIN_RATE)
pred_lasso_test = model_lasso.predict(X_test_std)
utils.plot_pred(y_test, pred_lasso_test, margin_rate=MARGIN_RATE)


# 線形SVR
C_SVR = 1 * 10 ** (-1)
EPSILON_SVR = 5 * 10 ** (-1)
model_svr = SVR(kernel="linear", C=C_SVR, epsilon=EPSILON_SVR, verbose=True)
model_svr.fit(X_train_std[:1000], y_train[:1000])
pred_svr_train = model_svr.predict(X_train_std)
utils.plot_pred(y_train, pred_svr_train, margin_rate=MARGIN_RATE)
pred_svr_test = model_svr.predict(X_test_std)
utils.plot_pred(y_test, pred_svr_test, margin_rate=MARGIN_RATE)



#
## lasso回帰(係数上位n位)
#N_COEF = 800
#coef_rank = np.argsort(np.abs(model_lasso.coef_))[::-1]
#model_lasso_new = Lasso(alpha=ALPHA_LASSO)
#model_lasso_new.fit(X_train_std[:,coef_rank[:N_COEF]], y_train)
#pred_lasso_new_train = model_lasso_new.predict(X_train_std[:,coef_rank[:N_COEF]])
#utils.plot_pred(y_train, pred_lasso_new_train, margin_rate=MARGIN_RATE)
#pred_lasso_new_test = model_lasso_new.predict(X_test_std[:,coef_rank[:N_COEF]])
#utils.plot_pred(y_test, pred_lasso_new_test, margin_rate=MARGIN_RATE)
#


