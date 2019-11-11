#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.spatial import distance


# 出力フォルダ作成関数
def make_new_folder(folder_name):
    if os.path.exists(folder_name)==False:
        os.mkdir(folder_name)
        return True
    else:
        return False
    
# 予測結果表示
def plot_pred(true, pred, margin_rate=0.1, s_rate=0.3, alpha=0.5):
    margin = (np.max(true) - np.min(true)) * margin_rate
    plt.plot(true, true, c="r", alpha=alpha)
    plt.scatter(true, pred, s=plt.rcParams['lines.markersize']**2 * s_rate, alpha=alpha)
    plt.title("r2 : "+str(r2_score(true, pred)))
    plt.grid()
    plt.xlim([np.min(true)-margin, np.max(true)+margin])
    plt.ylim([np.min(true)-margin, np.max(true)+margin])
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.show()


#離散判定
def check_discrete(df_data, tol=10):
    # ユニーク数tol以下を離散とする
    return (df_data.nunique() <= tol) & (df_data.dtypes!="float64")


# 四分位範囲が0より大きい連続データを取得
def check_large_q_cont(df_data, upper_q=.75, lower_q=.25, comparison="excess"):
    # 連続データを識別
    is_continuous = df_data.dtypes=="float64"
        
    # 四分位範囲を計算
    quantile_ranges = df_data.quantile(upper_q) - df_data.quantile(lower_q)
    
    # 四分位範囲が0より大きいかを識別
    if comparison=="excess":
        mask = (quantile_ranges > 0)  &  is_continuous
    else:
        mask = (quantile_ranges == 0) &  is_continuous
    
    #return df_data_cont.columns[mask]
    return mask


def _calc_mah_dis(y, x):
    tmp_df = pd.DataFrame(np.hstack((y, x)))
    mean   = np.mean(tmp_df, axis=0)
    cov    = np.cov(tmp_df.T)
    mah_dis = tmp_df.apply(distance.mahalanobis, v=mean, VI=np.linalg.pinv(cov), axis=1)
    return mah_dis

# 各特徴量と目的変数とのマハラノビス距離計算
def make_df_mah_dis(y_train, X_train):
    # 標準化
    y_train_str = StandardScaler().fit_transform(y_train)
    X_train_str = StandardScaler().fit_transform(X_train)
    
    #start = time.time()
    #df_mah_dis = pd.concat([_calc_mah_dis(y_train_str, X_train_str[:,[i]]) for i in range(100)], axis=1)
    #elapsed_time = time.time() - start
    #print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    
    # 各特徴量と目的変数とのマハラノビス距離計算
    df_mah_dis = pd.concat([_calc_mah_dis(y_train_str, X_train_str[:,[i]]) for i in range(X_train_str.shape[1])], axis=1)

    return df_mah_dis