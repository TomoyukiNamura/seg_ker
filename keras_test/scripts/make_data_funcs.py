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

def priorRawData(df_raw, tol_sigma = 2.5, window=7, min_periods=1, center=False):
    
    # 各キロ程の変化時期を特定
    #df_tmp_diff = org_dict["raw0_prior_treated"].dropna().diff().dropna()
    df_result = deepcopy(df_raw)
    df_tmp_diff = deepcopy(df_raw.dropna().diff().dropna())
    
    df_is_transition_dict = {}
    print("変化時期を特定")
    time.sleep(0.5)
    for milage in tqdm(list(df_tmp_diff.columns)):
        tmp_diff = np.array(df_tmp_diff[milage])
        
        mu = np.mean(tmp_diff)
        sigma = np.std(tmp_diff)
        
        df_is_transition_dict[milage] = list(df_tmp_diff[milage].index[np.abs(df_tmp_diff[milage])>mu+sigma*tol_sigma])
    
    
    # 各キロ程で変化時期別に移動平均を求める
    print("各キロ程で変化時期別に移動平均を求める")
    time.sleep(0.5)
    for milage in tqdm(list(df_result.columns)):
        # 対象の原系列取得
        #tmp_raw = org_dict["raw0_prior_treated"].loc[:,[milage]]
        tmp_raw = df_result.loc[:,[milage]]
        
        # 対象の原系列の変化情報を取得し，先頭に0，末尾にindexの最大値を置く
        tmp_transition_id = deepcopy(df_is_transition_dict[milage])
        tmp_transition_id.insert(0, 0)
        tmp_transition_id.append(max(list(tmp_raw.index)))
        
        # グループごとに移動平均を取得
        for i in range(len(tmp_transition_id)-1):
            tmp_raw_local = tmp_raw.iloc[range(tmp_transition_id[i], tmp_transition_id[i+1]),:]
            tmp_raw.iloc[range(tmp_transition_id[i], tmp_transition_id[i+1]),:] = tmp_raw_local.rolling(window=window, min_periods=min_periods, center=center).mean()
            
#            tmp_raw.iloc[range(tmp_transition_id[i], tmp_transition_id[i+1]),:] = tmp_raw_local.ewm(span=window).mean()
            
        # 移動平均結果を代入
        #org_dict["raw0_prior_treated"].loc[:,[milage]] = tmp_raw
        df_result.loc[:,[milage]] = tmp_raw
      
    return df_result


def priorDiffData(df_raw, n_diff, tol_sigma, window=7, min_periods=1, center=False):
    
    df_result = deepcopy(df_raw.diff())

    print("各キロ程で外れ値をNaNに変更")
    time.sleep(0.5)
    for milage in tqdm(list(df_result.columns)):
        # 差分系列取得
        #tmp_irregularity = np.array(org_dict["diff0"][milage].dropna())
        tmp_irregularity = np.array(df_result[milage].dropna())
        
        # 差分系列のmu, sigma計算
        mu = np.mean(tmp_irregularity)
        sigma = np.std(tmp_irregularity)
        
        # 差分系列の絶対値がmu + sigma*tol_sigmaを超過しているか調査
        #tmp_bool = np.abs(org_dict["diff0"][milage]) > mu + sigma*tol_sigma_diff_prior
        tmp_bool = np.abs(df_result[milage]) > mu + sigma*tol_sigma
        
        # mu + sigma*tol_sigmaを超過しているデータをNaNに変更
        #org_dict["diff0"][milage] = org_dict["diff0"][milage].where(tmp_bool==False)
        df_result[milage] = df_result[milage].where(tmp_bool==False)
        
    # 移動平均
    df_result = df_result.rolling(window=window, min_periods=min_periods, center=center).mean()
        
    return df_result