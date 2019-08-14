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


from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA


# 設定
DATA_NAME   = "valid2"
RANDOM_SEED = 0
TEST_SIZE   = 0.5
MARGIN_RATE = 0.01


# データ読み込み
df_train = pd.read_csv(f"data/{DATA_NAME}/train.csv")
df_train.shape
df_train.columns

df_test = pd.read_csv(f"data/{DATA_NAME}/test.csv")
df_test.shape
df_test.columns

#df_mah_dis = pd.read_csv(f"data/{DATA_NAME}/train_mah_dis.csv")
#df_mah_dis.columns = df_train.iloc[:,2:].columns


# 訓練・評価データ変換
X_train = df_train.iloc[:, 2:df_train.shape[1]]
y_train = df_train.iloc[:, [1]]
X_test  = df_test.iloc[:, 2:df_test.shape[1]]
y_test  = df_test.iloc[:, [1]]


# # 四分位範囲が0より大きい連続データを取得
large_q_cont_columns = utils.check_large_q_cont(X_train, upper_q=.75, lower_q=.25, comparison="excess")




zero_q_cont_columns = utils.check_large_q_cont(X_train, upper_q=.75, lower_q=.25, comparison="equal")




tmp_X_train = X_train[zero_q_cont_columns]


#mask = (tmp_X_train == tmp_X_train.median()).all(axis=1)
#mask[mask]

mask = (tmp_X_train == tmp_X_train.median()).sum(axis=1) / tmp_X_train.shape[1] > 0.98


X_train = X_train.loc[mask==False,:]
y_train = y_train.loc[mask==False,:]


#RATE_REMOVE = 0.005
##df_mah_dis = pd.DataFrame(StandardScaler().fit_transform(df_mah_dis))
#df_mah_dis_sum = df_mah_dis.loc[:,large_q_cont_columns].sum(axis=1)
#
#tol_remove = df_mah_dis_sum.quantile(1-RATE_REMOVE)
#mask = df_mah_dis_sum < tol_remove
#
#X_train = X_train.loc[mask,:]
#y_train = y_train.loc[mask,:]


## マハラノビス距離付きプロット
#col_num = 600
#X_train_1 = X_train.iloc[:,[col_num]]
#X_train_1.describe()
#cmap = plt.get_cmap("Blues")
#plt.scatter(X_train_1, y_train, s=plt.rcParams['lines.markersize']**2 * 0.3, alpha=1,
#            c=cmap(df_mah_dis.iloc[:,col_num]/3))



#
## 標準化
##X_train = StandardScaler().fit_transform(X_train)
#
#
#X_train_1 = X_train.iloc[:,[1650]]
#X_train_1.describe()
#plt.scatter(X_train_1, y_train, s=plt.rcParams['lines.markersize']**2 * 0.3, alpha=0.5)
#
## kNN
#NK = 5
#neigh = NearestNeighbors(n_neighbors=NK)
#tmp_df = pd.concat([y_train, X_train_1], axis=1)
#neigh.fit(tmp_df)
#d = neigh.kneighbors(tmp_df)[0]
#cmap = plt.get_cmap("Blues")
#plt.scatter(X_train_1, y_train, s=plt.rcParams['lines.markersize']**2 * 0.3, alpha=1,
#            c=cmap(np.sum(d, axis=1)))
#
#
## マハラノビス距離
#tmp_df = pd.concat([y_train, X_train_1], axis=1)
#mean = np.mean(tmp_df, axis=0)
#cov  = np.cov(tmp_df.T)
#mah_dis = tmp_df.apply(distance.mahalanobis, v=mean, VI=np.linalg.pinv(cov), axis=1)
#plt.scatter(X_train_1, y_train, s=plt.rcParams['lines.markersize']**2 * 0.3, alpha=1,
#            c=cmap(mah_dis/5.5))






# 標準化
std_scaler = StandardScaler()
#std_scaler = RobustScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std  = std_scaler.transform(X_test)
X_train_post = X_train_std
X_test_post  = X_test_std


#主成分分析の実行
#pca = PCA()
#X_train_feature = pca.fit_transform(X_train_std)
#X_test_feature  = pca.transform(X_test_std)
#X_train_post = X_train_feature[:,:1000]
#X_test_post  = X_test_feature[:,:1000]

# yについてnumpyへ変換
y_train = y_train.values
y_test  = y_test.values



## 線形回帰
#model_lr    = LinearRegression()
#model_lr.fit(X_train_post, y_train)
#pred_lr_train    = model_lr.predict(X_train_post)
#utils.plot_pred(y_train, pred_lr_train, margin_rate=MARGIN_RATE)
#pred_lr_test    = model_lr.predict(X_test_post)
#utils.plot_pred(y_test, pred_lr_test, margin_rate=MARGIN_RATE)


## GradientBoostingRegressor
#model_gbr = GradientBoostingRegressor(max_depth=10)
#model_gbr.fit(X_train_post, y_train)
##pred_gbr_train = model_gbr.predict(X_train_post)
##utils.plot_pred(y_train, pred_gbr_train, margin_rate=MARGIN_RATE)
#pred_gbr_test = model_gbr.predict(X_test_post)
#utils.plot_pred(y_test, pred_gbr_test, margin_rate=MARGIN_RATE)
#
## knn
#N_NEIGHBORS = 5
#model_knn = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
#model_knn.fit(X_train_post, y_train)
##pred_knn_train = model_knn.predict(X_train_post)
##utils.plot_pred(y_train, pred_knn_train, margin_rate=MARGIN_RATE)
#pred_knn_test = model_knn.predict(X_test_post)
#utils.plot_pred(y_test, pred_knn_test, margin_rate=MARGIN_RATE)


# リッジ回帰
ALPHA_RIDGE = 1 * 10 ** 2.5
model_ridge = Ridge(alpha= ALPHA_RIDGE)
model_ridge.fit(X_train_post, y_train)
#pred_ridge_train = model_ridge.predict(X_train_post)
#utils.plot_pred(y_train, pred_ridge_train, margin_rate=MARGIN_RATE)
pred_ridge_test = model_ridge.predict(X_test_post)
utils.plot_pred(y_test, pred_ridge_test, margin_rate=MARGIN_RATE, alpha=0.2)


# lasso回帰
ALPHA_LASSO = 3 * 10 ** (-3)
#ALPHA_LASSO = 3 * 10 ** (-3)
model_lasso = Lasso(alpha=ALPHA_LASSO)
model_lasso.fit(X_train_post, y_train)
#pred_lasso_train = model_lasso.predict(X_train_post)
#utils.plot_pred(y_train, pred_lasso_train, margin_rate=MARGIN_RATE)
pred_lasso_test = model_lasso.predict(X_test_post)
utils.plot_pred(y_test, pred_lasso_test, margin_rate=MARGIN_RATE)


# 線形SVR
C_SVR = 1 * 10 ** 1
EPSILON_SVR = 1 * 10 ** (-1)
model_svr = SVR(kernel="linear", C=C_SVR, epsilon=EPSILON_SVR, verbose=True)
model_svr.fit(X_train_post[:3000], y_train[:3000])
#pred_svr_train = model_svr.predict(X_train_post)
#utils.plot_pred(y_train, pred_svr_train, margin_rate=MARGIN_RATE)
pred_svr_test = model_svr.predict(X_test_post)
utils.plot_pred(y_test, pred_svr_test, margin_rate=MARGIN_RATE)



#
## lasso回帰(係数上位n位)
#N_COEF = 800
#coef_rank = np.argsort(np.abs(model_lasso.coef_))[::-1]
#model_lasso_new = Lasso(alpha=ALPHA_LASSO)
#model_lasso_new.fit(X_train_post[:,coef_rank[:N_COEF]], y_train)
#pred_lasso_new_train = model_lasso_new.predict(X_train_post[:,coef_rank[:N_COEF]])
#utils.plot_pred(y_train, pred_lasso_new_train, margin_rate=MARGIN_RATE)
#pred_lasso_new_test = model_lasso_new.predict(X_test_post[:,coef_rank[:N_COEF]])
#utils.plot_pred(y_test, pred_lasso_new_test, margin_rate=MARGIN_RATE)
#


