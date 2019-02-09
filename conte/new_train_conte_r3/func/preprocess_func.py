#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

Preprocessingスクリプト用関数 
 
"""

import os
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import time

# 出力フォルダ作成
def makeNewFolder(folder_name):
    if os.path.exists(folder_name)==False:
        os.mkdir(folder_name)
        return True
    else:
        return False

def priorRawData(df_raw, window, min_periods, center, tol_diff=0.7, tol_n_group=5, window_median=20):
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
        displacement_point_list.append(max(list(df_result[milage].index))+1)
        
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
    df_result = df_result.rolling(window=window_median, min_periods=1, center=True).median()
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


def makeStartRawDict(df_raw, start_date_id, start_period, n_average_date, start_average_method):
    print("\n・原データ初期値を取得")
    time.sleep(0.5)
    
    start_raw_dict = {}
    
    for milage in tqdm(list(df_raw.columns)):
        # 対象期間の原系列を取得
        tmp_raw = df_raw.loc[range(start_date_id-start_period+1,start_date_id+1),milage]
            
        # nan除く過去n_start_date日分を初期値が計算対象
        start_vector = tmp_raw.dropna().tail(n_average_date)
        
        ## ひとまずお蔵入り ======================================
    #    # start_vectorの長さが0の場合，nan以外全期間のうち過去n_start_date日分を初期値を計算対象とする
    #    if start_vector.shape[0]==0:
    #        start_vector = df_raw.loc[range(start_date_id+1),milage].dropna().tail(n_average_date)
        ## ひとまずお蔵入り ======================================
        
        # start_vectorの長さにより分岐
        if start_vector.shape[0]>0:
            # start_vectorの長さが0以上の場合，代表値を計算
            if start_average_method=="mean":
                start_raw_dict[milage] = np.mean(start_vector)
            else:
                start_raw_dict[milage] = np.median(start_vector)
            
        else:
            start_raw_dict[milage] = np.nan
        
    # Series化
    start_raw_dict = pd.Series(start_raw_dict)
    
    # nanを直線補完
    start_raw_dict = start_raw_dict.interpolate()
    
    # 辞書型化
    start_raw_dict = start_raw_dict.to_dict()
    
    return start_raw_dict


def makeStartDiffDict(df_dict, n_diff, start_date_id):
    print("\n・差分初期値を取得")
    time.sleep(0.5)
    
    start_diff_dict = {}
    start_values_result_dict = {}
    
    for milage in tqdm(list(df_dict[f"diff0"].columns)):
        # 差分系列(NaNの場合は0で補完)
        n_diff_iszero = 0
        start_diff = []
    
        for i in range(n_diff):
            tmp_data = df_dict[f"diff{i}"][milage][start_date_id]
            
            if np.isnan(tmp_data):
                n_diff_iszero += 1
                start_diff.append(np.array(0))
            else:
                start_diff.append(np.array(tmp_data))
                
        start_diff_dict[milage] = np.dstack(start_diff)[0]
        
        # 初期値作成の結果をまとめる
        start_values_result_dict[milage] = {"n_diff_iszero":n_diff_iszero}
    
    return start_diff_dict, start_values_result_dict


## 以下，新しく追加した関数 ===============================

def checkOrgTrainNum(df_irregularity_phase_modified):
    print("\n・直近180日間の原系列にどれほどデータ数があるか===============================")
    time.sleep(0.5)
    n_org_train_dict = {}
    n_org = df_irregularity_phase_modified.shape[0]
    tmp_bool = df_irregularity_phase_modified.iloc[list(range(n_org-180, n_org)),:].isna()==False
    for milage in tqdm(list(df_irregularity_phase_modified.columns)):
        n_org_train_dict[milage] = tmp_bool[milage][tmp_bool[milage]==True].shape[0]
        
    return n_org_train_dict
    

## ARIMA(n_diff,1,0)の訓練データ作成
def dfDict2ARIMAInput(df_dict, n_diff):
    print("\n・モデルに入力可能な形式にデータを変換")
    time.sleep(0.5)
    
    train_data_dict = {}
    
    for milage in tqdm(list(df_dict[f"diff0"].columns)):
        
        # nanを除去
        df_y_X = []
        for i in range(n_diff+1):
            df_y_X.append(df_dict[f"diff{i}"].loc[:,[milage]])
        df_y_X = pd.concat(df_y_X, axis=1).dropna()
        
        # 目的変数(0時点差分データ)を作成
        y = np.array(df_y_X.iloc[:,[0]])
        
        # 説明変数(-n_diff ~ -1時点差分データ)を作成
        X = []
        for i in range(1,n_diff+1):
            X.append(np.array(df_y_X.iloc[:,i]))
        X = np.dstack(X)[0]
        
        train_data_dict[milage] = (y ,X)
        
    return train_data_dict


def makeTimeSpaceData(input_file):
    print("\n・行：日付id, 列：キロ程　のデータ作成")
    time.sleep(0.5)
    
    # データ読み込み
    df_track = pd.read_csv(input_file)
    df_track.head()
    
    # 対象カラム名
    columns = "高低左"
    
    #キロ程リスト
    milage_list = list(df_track.loc[:,"キロ程"].unique())
    
    df_irregularity = []
    for milage in tqdm(milage_list):
        # 対象キロ程のデータを取得
        tmp_df = df_track[df_track.loc[:,"キロ程"]==milage].loc[:,[columns]]
        
        # インデックスを初期化
        tmp_df = tmp_df.reset_index(drop=True)
        
        # リネームしてアペンド
        df_irregularity.append(tmp_df.rename(columns={columns: f"m{milage}"}))
    
    df_irregularity = pd.concat(df_irregularity, axis=1)
    
    return df_irregularity



#def phaseModify(df ,n_shift, n_sub_data):
#    print("\n・位相ずれを修正")
#    time.sleep(0.5)
#    
#    date_list = list(df.columns)
#        
#    # 分割点取得
#    devide_point = list(np.arange(0, df.shape[0], n_sub_data))
#    devide_point.append(df.shape[0])
#    
#    
#    # 基準キロ程取得
#    reference_date = df.loc[:,date_list[0]]
#    
#    for date_id in range(1,len(date_list)):
#        #  補正対象キロ程取得
#        target_date = df.loc[:,date_list[date_id]]
#        
#        # 補正後キロ程の保存場所
#        target_date_modified = []
#        
#        print(f"\n{date_id}_{date_list[date_id]}")
#        time.sleep(0.3)
#        
#        for dp_id in tqdm(range(len(devide_point)-1)):
#            # 現在区間の基準キロ程，補正対象キロ程取得
#            tmp_reference = reference_date.loc[range(devide_point[dp_id], devide_point[dp_id+1]),]
#            tmp_target = target_date.loc[range(devide_point[dp_id], devide_point[dp_id+1]),]
#            
#            # データ数取得
#            n_tmp_reference = tmp_reference[tmp_reference.isna()==False].shape[0]
#            n_tmp_target = tmp_target[tmp_target.isna()==False].shape[0]
#            
#            # 両方とも10%以上ある場合，補正をかける
#            if n_tmp_reference >= n_sub_data*0.1 and n_tmp_target >= n_sub_data*0.1:
#                
#                # 
#                tmp_corr_dict = {}
#                tmp_n_dict = {}
#                
#                for shift in range(-n_shift, n_shift+1):
#                    # not naの取得
#                    na_tmp_reference = tmp_reference.isna()
#                    na_tmp_target = tmp_target.shift(shift).isna()
#                    not_na = (na_tmp_reference | na_tmp_target) == False
#                    
#                    # 相関係数計算のためのデータ数取得
#                    tmp_n_dict[shift] = tmp_reference[not_na].shape[0]
#                    
#                    # 相関係数計算
#                    tmp_corr_dict[shift] = tmp_reference[not_na].corr(tmp_target.shift(shift)[not_na])
#                    
#                shift_array = np.array(list(tmp_corr_dict.keys()))
#                corr_array = np.array(list(tmp_corr_dict.values()))
#                n_array = np.array(list(tmp_n_dict.values()))
#                
#                # corr_arrayがnan以外，またはn_arrayが n_sub_data*0.1以上のshift_arrayとcorr_arrayを取得
#                shift_array = shift_array[np.logical_or(np.isnan(corr_array)==False, n_array>=n_sub_data*0.1)]
#                corr_array = corr_array[np.logical_or(np.isnan(corr_array)==False, n_array>=n_sub_data*0.1)]
#                
#                if len(corr_array)>0:
#                    # 相関係数が最大となる補正値を取得
#                    shift_max_corr = shift_array[corr_array==np.max(corr_array)][0]
#                    
#                    # 補正結果を保存
#                    target_date_modified.append(deepcopy(target_date.shift(shift_max_corr).loc[range(devide_point[dp_id], devide_point[dp_id+1]),]))
#                    
#                else:
#                    target_date_modified.append(deepcopy(tmp_target))
#                
#            else:
#                target_date_modified.append(deepcopy(tmp_target))
#    
#        # 補正結果をpd.Series化
#        target_date_modified = pd.concat(target_date_modified, axis=0)
#        target_date_modified.index = target_date.index
#        
#        # 補正結果を置き換え
#        df.loc[:,date_list[date_id]] = target_date_modified
#    
#    # 列がキロ程になるよう転置
#    df = df.T
#    
#    return df
    


def savePreprocessData(milage_list, train_max_min, init_raw_dict, init_diff_dict, training_data_dict, n_org_train_dict, output_path):
    print("\n・訓練データ，初期値を保存")
    time.sleep(0.5)
    
    # キロ程リスト
    np.savez(f"{output_path}/milage_list.npz", milage_list)
    
    # 
    np.savez(f"{output_path}/train_max_min.npz", train_max_min)
    
    #
    for milage in tqdm(milage_list):
        # アウトプットフォルダを作成
        tmp_output_path = f"{output_path}/{milage}"
        if os.path.exists(tmp_output_path)==False: os.mkdir(tmp_output_path)
        
        # 初期原データを保存
        np.savez(f"{tmp_output_path}/init_raw.npz", init_raw_dict[milage])
        
        # 初期差分を保存
        np.savez(f"{tmp_output_path}/init_diff.npz", init_diff_dict[milage])
        
        # 訓練データを保存
        np.savez(f"{tmp_output_path}/train_data.npz", training_data_dict[milage][0], training_data_dict[milage][1])
        
        # 直近180日間の原系列のデータ数を保存
        np.savez(f"{tmp_output_path}/n_orgdata.npz", n_org_train_dict[milage])