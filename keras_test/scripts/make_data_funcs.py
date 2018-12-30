#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import time


def priorRawData(df_raw, window, min_periods, center, tol_diff=0.7, tol_n_group=5):
    # 入力した原系列をコピー
    df_result = deepcopy(df_raw)
    
    # メディアンフィルタ
    tmp_raw0_median = df_result.rolling(window=20, min_periods=1, center=True).median()
    
    # 10日間の差分
    tmp_raw0_median_diff = np.abs(tmp_raw0_median.diff(10))
    
    # 差分がtol_diffより大きいところを取得
#    tmp_raw0_median_diff_over_tol_diff = tmp_raw0_median_diff > tol_diff
    tmp_raw0_median_diff_over_tol_diff = np.logical_or(tmp_raw0_median_diff > tol_diff, tmp_raw0_median_diff.isna())
    
    # キロ程別に差分が1より大きいindexを取得
    for milage in tqdm(list(df_result.columns)):
        # 10日間の差分がtol_diffより大きいidを取得
        over_tol_diff_id = pd.Series(list(tmp_raw0_median_diff[milage][tmp_raw0_median_diff_over_tol_diff[milage]].index))
        over_tol_diff_id = over_tol_diff_id.iloc[range(10,over_tol_diff_id.shape[0])].reset_index(drop=True)        
        
        # indexが連続しているところでグループ化
        tmp_diff = over_tol_diff_id.diff()
        tmp_diff[0] = 0.0
        
        group_list = []
        tmp_group_list = []
        for i in range(over_tol_diff_id.shape[0]):
            if tmp_diff[i] != 1.0 and i!=0:
                group_list.append(tmp_group_list)
                tmp_group_list = []
            tmp_group_list.append(over_tol_diff_id[i])
        group_list.append(tmp_group_list)
        
        
        # 変位点を取得（グループ内のデータ数がtol_n_groupより大きいもののみ）
        displacement_point_list = []
        for i in range(len(group_list)):
            if len(group_list[i]) >= tol_n_group:
                displacement_point_list.append(group_list[i][0])
                
                
        # 変位点前後のデータの足並みを揃える
        displacement_point_list.insert(0, 0)
        displacement_point_list.append(max(list(df_result[milage].index)))
        
        tmp_raw = df_result.loc[:,[milage]]
            
        for i in range(len(displacement_point_list)-2):
            # グループ1,2を取得
            tmp_raw_local_1 = tmp_raw.iloc[range(displacement_point_list[i], displacement_point_list[i+1]),:]
            tmp_raw_local_2 = tmp_raw.iloc[range(displacement_point_list[i+1], displacement_point_list[i+2]),:]
            
            if len(tmp_raw_local_1.dropna())!=0 and len(tmp_raw_local_2.dropna())!=0:
                
                # グループ1の末尾3平均，グループ2の先頭3平均を算出
                mean_1 = float(np.median(tmp_raw_local_1.dropna().tail(20)))
                mean_2 = float(np.median(tmp_raw_local_2.dropna().head(20)))
                            
                # 平均差をグループ2に足し，保存
                tmp_raw.iloc[range(displacement_point_list[i+1], displacement_point_list[i+2]),:] = (mean_1 - mean_2) + tmp_raw_local_2
        
        # 移動平均結果を代入
        df_result.loc[:,[milage]] = tmp_raw
    
    # メディアンフィルタ
    df_result = df_result.rolling(window=20, min_periods=1, center=True).median()
    df_result = df_result.rolling(window=window, min_periods=min_periods, center=center).mean()
    
    return df_result, tmp_raw0_median, tmp_raw0_median_diff

def priorDiffData(org_df_raw, df_raw, n_diff, tol_sigma, window=7, min_periods=1, center=False):
    
    df_result = deepcopy(df_raw.diff())

    print("各キロ程で外れ値をNaNに変更")
    time.sleep(0.5)
    for milage in tqdm(list(df_result.columns)):
        # 差分系列取得
        tmp_irregularity = df_result[milage].dropna()
        
        # 前処理前の差分系列から，もともとNaNでない部分の情報をちゅうしゅつ
        isnot_nan = org_df_raw[milage].diff().isna()==False
        
        # もともとNaNでない部分に限定し，25%点，75％点を計算
        Q3 = tmp_irregularity[isnot_nan].quantile(.75)
        Q1 = tmp_irregularity[isnot_nan].quantile(.25)
        
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