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
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.decomposition import PCA,KernelPCA


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


# 訓練・評価データ変換
X_train = df_train.iloc[:, 2:df_train.shape[1]]
y_train = df_train.iloc[:, [1]].values
X_test  = df_test.iloc[:, 2:df_test.shape[1]]
y_test  = df_test.iloc[:, [1]].values


## 
# 四分位範囲が0より大きい連続データ
is_non_range_cont  = utils.check_large_q_cont(X_train, upper_q=.75, lower_q=.25, comparison="equal")
X_train_1 = X_train.loc[:, is_non_range_cont]
X_test_1  = X_test.loc[:, is_non_range_cont]

# 離散データ
is_discrete = utils.check_discrete(X_train, tol=10000)
X_train_2   = X_train.loc[:, is_discrete]
X_test_2    = X_test.loc[:, is_discrete]

# その他(一般の連続データ)
is_cont = (is_non_range_cont==False) & (is_discrete==False)
X_train_3 = X_train.loc[:, is_cont]
X_test_3  = X_test.loc[:, is_cont]



## その他(一般の連続データ)を主成分分析
#pca = PCA()
#X_train_3_feature = pca.fit_transform(X_train_3)
#X_test_3_feature  = pca.transform(X_test_3)
#X_train_3_pca     = pd.DataFrame(X_train_3_feature[:,:2])
#X_test_3_pca      = pd.DataFrame(X_test_3_feature[:,:2])

#kpca = KernelPCA(kernel="rbf", fit_inverse_transform=False, gamma=10)
#X_train_3_feature = kpca.fit_transform(X_train_3.iloc[:,0:100])
#X_test_3_feature  = kpca.transform(X_test_3.iloc[:,0:100])
#
#X_train_3_pca     = pd.DataFrame(X_train_3_feature[:,:5])
#X_test_3_pca      = pd.DataFrame(X_test_3_feature[:,:5])



#
#
#import matplotlib.ticker as ticker
#plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
#plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_[:6])), "-o")
#plt.xlabel("Number of principal components")
#plt.ylabel("Cumulative contribution rate")
#plt.grid()
#plt.show()



# 説明変数をまとめる 
#X_train_post = pd.concat([X_train_1, X_train_2, X_train_3_pca],axis=1)
#X_test_post  = pd.concat([X_test_1, X_test_2, X_test_3_pca],axis=1)
X_train_post = pd.concat([X_train_1, X_train_2],axis=1)
X_test_post  = pd.concat([X_test_1, X_test_2],axis=1)
#X_train_post = pd.concat([X_train_1, X_train_3.iloc[:,0:300]],axis=1)
#X_test_post  = pd.concat([X_test_1, X_test_3.iloc[:,0:300]],axis=1)

#X_train_post = pd.concat([X_train_3.iloc[:,0:300]],axis=1)
#X_test_post  = pd.concat([X_test_3.iloc[:,0:300]],axis=1)


# 説明変数を標準化
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train_post)
X_test_std  = std_scaler.transform(X_test_post)


# RandomForestRegressor
model_rf = RandomForestRegressor(max_depth=20,random_state=RANDOM_SEED)
model_rf.fit(X_train_std, y_train)
pred_rf_test = model_rf.predict(X_test_std)
utils.plot_pred(y_test, pred_rf_test, margin_rate=MARGIN_RATE)



# knn
N_NEIGHBORS = 10
model_knn = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
model_knn.fit(X_train_std, y_train)
pred_knn_test = model_knn.predict(X_test_std)
utils.plot_pred(y_test, pred_knn_test, margin_rate=MARGIN_RATE)



# リッジ回帰
ALPHA_RIDGE = 1 * 10 ** 0.000001
model_ridge = Ridge(alpha= ALPHA_RIDGE)
model_ridge.fit(X_train_post, y_train)
pred_ridge_test = model_ridge.predict(X_test_post)
utils.plot_pred(y_test, pred_ridge_test, margin_rate=MARGIN_RATE, alpha=0.3)

## SVR
#C_SVR = 1 * 10 ** 1
#EPSILON_SVR = 1 * 10 ** (-1)
#model_svr = SVR(kernel="rbf", C=C_SVR, epsilon=EPSILON_SVR, verbose=True)
#model_svr.fit(X_train_std, y_train)
#pred_svr_test = model_svr.predict(X_test_std)
#utils.plot_pred(y_test, pred_svr_test, margin_rate=MARGIN_RATE)