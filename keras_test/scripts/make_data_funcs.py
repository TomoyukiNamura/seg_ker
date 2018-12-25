#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import mean_absolute_error

def priorRawData(df_raw, tol_sigma = 2.5, window=7, min_periods=1, center=False, detect_outlier_method = "quantile"):
    
    # 入力した原系列をコピー
    df_result = deepcopy(df_raw)
     
    # 差分系列を計算(nan除外)
    df_tmp_diff = deepcopy(df_raw.rolling(window=5, min_periods=1, center=True).median().dropna().diff().dropna())
    
#    # メディアンフィルタ前後のプロット
#    plt.plot(df_result[milage])
#    plt.plot(df_result.rolling(window=5, min_periods=1, center=True).median()[milage])

    df_is_transition_dict = {}
    print("変化時期を特定し，各キロ程で変化時期別に移動平均を求める")
    time.sleep(0.5)
    for milage in tqdm(list(df_tmp_diff.columns)):
        # 対象の差分系列を取得
        tmp_diff = df_tmp_diff[milage]
                
        if detect_outlier_method == "mean_sigma":
            # 差分系列の平均，標準偏差を取得
            mu = np.mean(tmp_diff)
            sigma = np.std(tmp_diff)
            
            # 差分系列の絶対値がmu+sigma*tol_sigmaを超過しているデータを外れ値とみなす
            is_outlier = np.abs(tmp_diff)>mu+sigma*tol_sigma
            
        elif detect_outlier_method == "quantile":
            # 25%点，75％点を計算
            Q3 = tmp_diff.quantile(.75)
            Q1 = tmp_diff.quantile(.25)
            
            # 4分位範囲計算
            quantile_range = (Q3 - Q1) * tol_sigma
            
            # Q1-4分位範囲未満，またはQ3+4分位範囲超過のデータを外れ値とする
            is_lower_outlier = tmp_diff < Q1-quantile_range
            is_upper_outlier = tmp_diff > Q3+quantile_range
            is_outlier = np.logical_or(is_lower_outlier, is_upper_outlier)
            
        
        # 外れ値としたデータのindexを保存
        df_is_transition_dict[milage] = list(df_tmp_diff[milage].index[is_outlier])


    print("各キロ程で変化時期をスケールを合わせて結合し，移動平均を求める")
    time.sleep(0.5)
    for milage in tqdm(list(df_tmp_diff.columns)):
        # 対象の原系列取得
        tmp_raw = deepcopy(df_result.loc[:,[milage]])
        
#        plt.plot(tmp_raw);plt.ylim([-5.5, -1.0]);plt.grid()
        
        # 対象の原系列の変化情報を取得
        tmp_transition_id = deepcopy(df_is_transition_dict[milage])
        tmp_transition_id.insert(0, 0)
        tmp_transition_id.append(max(list(tmp_raw.index)))
        
        # グループを調査し，NaNだけのグループがあれば除外
        remove_list = []
        for i in range(len(tmp_transition_id)-1):
            tmp_raw_local = tmp_raw.iloc[range(tmp_transition_id[i], tmp_transition_id[i+1]),:]
            
            if tmp_raw_local.isna().all()[milage]:
                remove_list.append(tmp_transition_id[i])

        for remove_id in remove_list:
            tmp_transition_id.remove(remove_id)
        
        # 変化時期をスケールを合わせて結合
        for i in range(len(tmp_transition_id)-2):
            # グループ1,2を取得
            tmp_raw_local_1 = tmp_raw.iloc[range(tmp_transition_id[i], tmp_transition_id[i+1]),:]
            tmp_raw_local_2 = tmp_raw.iloc[range(tmp_transition_id[i+1], tmp_transition_id[i+2]),:]
        
            # グループ1の末尾3平均，グループ2の先頭3平均を算出
            mean_1 = float(np.mean(tmp_raw_local_1.dropna().tail(10)))
            mean_2 = float(np.mean(tmp_raw_local_2.dropna().head(10)))
                        
            # 平均差をグループ2に足し，保存
            tmp_raw.iloc[range(tmp_transition_id[i+1], tmp_transition_id[i+2]),:] = (mean_1 - mean_2) + tmp_raw_local_2
        
#        plt.plot(tmp_raw);plt.ylim([-5.5, -1.0]);plt.grid();plt.show()
        
        # 移動平均実施
        tmp_raw = tmp_raw.rolling(window=window, min_periods=min_periods, center=center).mean()
        #tmp_raw = tmp_raw.ewm(span=window, min_periods=min_periods).mean()
        
#        # グループごとに移動平均を取得
#        for i in range(len(tmp_transition_id)-1):
#            tmp_raw_local = tmp_raw.iloc[range(tmp_transition_id[i], tmp_transition_id[i+1]),:]
#            tmp_raw.iloc[range(tmp_transition_id[i], tmp_transition_id[i+1]),:] = tmp_raw_local.rolling(window=window, min_periods=min_periods, center=center).mean()
#                        
        # 移動平均結果を代入
        df_result.loc[:,[milage]] = tmp_raw    
    
#    print("各キロ程で変化時期別に移動平均を求める")
#    time.sleep(0.5)
#    for milage in tqdm(list(df_tmp_diff.columns)):
#        # 対象の原系列取得
#        tmp_raw = df_result.loc[:,[milage]]
#        
#        # 対象の原系列の変化情報を取得し，先頭に0，末尾にindexの最大値を置く
#        tmp_transition_id = deepcopy(df_is_transition_dict[milage])
#        tmp_transition_id.insert(0, 0)
#        tmp_transition_id.append(max(list(tmp_raw.index)))
#        
#        # グループごとに移動平均を取得
#        for i in range(len(tmp_transition_id)-1):
#            tmp_raw_local = tmp_raw.iloc[range(tmp_transition_id[i], tmp_transition_id[i+1]),:]
#            tmp_raw.iloc[range(tmp_transition_id[i], tmp_transition_id[i+1]),:] = tmp_raw_local.rolling(window=window, min_periods=min_periods, center=center).mean()
#                        
#        # 移動平均結果を代入
#        df_result.loc[:,[milage]] = tmp_raw
      
    return df_result


def priorDiffData(df_raw, n_diff, tol_sigma, window=7, min_periods=1, center=False):
    
    df_result = deepcopy(df_raw.diff())

    print("各キロ程で外れ値をNaNに変更")
    time.sleep(0.5)
    for milage in tqdm(list(df_result.columns)):
        # 差分系列取得
        tmp_irregularity = df_result[milage].dropna()
        
        # 25%点，75％点を計算
        Q3 = tmp_irregularity.quantile(.75)
        Q1 = tmp_irregularity.quantile(.25)
        
        # 4分位範囲計算
        quantile_range = (Q3 - Q1) * tol_sigma
        
        
        # Q1-4分位範囲未満，またはQ3+4分位範囲超過のデータを外れ値とする
        tmp_is_lower_outlier = df_result[milage] < Q1-quantile_range
        tmp_is_upper_outlier = df_result[milage] > Q3+quantile_range
        tmp_is_outlier = np.logical_or(tmp_is_lower_outlier, tmp_is_upper_outlier)
        
        # 外れ値をNaNに変更
        df_result[milage] = df_result[milage].where(tmp_is_outlier==False)
        
        ## 4σによる外れ値検出（小さい外れ値が残るためお蔵入り） ============
        
#        # 差分系列のmu, sigma計算
#        mu = np.mean(tmp_irregularity)
#        sigma = np.std(tmp_irregularity)
#        
#        # 差分系列の絶対値がmu + sigma*tol_sigmaを超過しているか調査
#        tmp_bool = np.abs(df_result[milage]) > mu + sigma*tol_sigma
        
#        # mu + sigma*tol_sigmaを超過しているデータをNaNに変更
#        df_result[milage] = df_result[milage].where(tmp_bool==False)
        
        ## 4σによる外れ値検出（小さい外れ値が残るためお蔵入り） ============
       
        
    # 移動平均
    df_result = df_result.rolling(window=window, min_periods=min_periods, center=center).mean()
        
    return df_result