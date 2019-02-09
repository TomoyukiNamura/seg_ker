#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
os.chdir("/Users/tomoyuki/Desktop/keras_test")

import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import time


def phaseModify(df ,n_shift, n_sub_data):
    date_list = list(df.columns)
        
    # 分割点取得
    devide_point = list(np.arange(0, df.shape[0], n_sub_data))
    devide_point.append(df.shape[0])
    
    
    # 基準キロ程取得
    reference_date = df.loc[:,date_list[0]]
    
    for date_id in range(1,len(date_list)):
        #  補正対象キロ程取得
        target_date = df.loc[:,date_list[date_id]]
        
        # 補正後キロ程の保存場所
        target_date_modified = []
        
        print(f"\n{date_id}_{date_list[date_id]}")
        time.sleep(0.3)
        
        for dp_id in tqdm(range(len(devide_point)-1)):
            # 現在区間の基準キロ程，補正対象キロ程取得
            tmp_reference = reference_date.loc[range(devide_point[dp_id], devide_point[dp_id+1]),]
            tmp_target = target_date.loc[range(devide_point[dp_id], devide_point[dp_id+1]),]
            
            # データ数取得
            n_tmp_reference = tmp_reference[tmp_reference.isna()==False].shape[0]
            n_tmp_target = tmp_target[tmp_target.isna()==False].shape[0]
            
            # 両方とも10%以上ある場合，補正をかける
            if n_tmp_reference >= n_sub_data*0.1 and n_tmp_target >= n_sub_data*0.1:
                
                # 
                tmp_corr_dict = {}
                tmp_n_dict = {}
                
                for shift in range(-n_shift, n_shift+1):
                    # not naの取得
                    na_tmp_reference = tmp_reference.isna()
                    na_tmp_target = tmp_target.shift(shift).isna()
                    not_na = (na_tmp_reference | na_tmp_target) == False
                    
                    # 相関係数計算のためのデータ数取得
                    tmp_n_dict[shift] = tmp_reference[not_na].shape[0]
                    
                    # 相関係数計算
                    tmp_corr_dict[shift] = tmp_reference[not_na].corr(tmp_target.shift(shift)[not_na])
                    
                shift_array = np.array(list(tmp_corr_dict.keys()))
                corr_array = np.array(list(tmp_corr_dict.values()))
                n_array = np.array(list(tmp_n_dict.values()))
                
                # corr_arrayがnan以外，またはn_arrayが n_sub_data*0.1以上のshift_arrayとcorr_arrayを取得
                shift_array = shift_array[np.logical_or(np.isnan(corr_array)==False, n_array>=n_sub_data*0.1)]
                corr_array = corr_array[np.logical_or(np.isnan(corr_array)==False, n_array>=n_sub_data*0.1)]
                
                if len(corr_array)>0:
                    # 相関係数が最大となる補正値を取得
                    shift_max_corr = shift_array[corr_array==np.max(corr_array)][0]
                    
                    # 補正結果を保存
                    target_date_modified.append(deepcopy(target_date.shift(shift_max_corr).loc[range(devide_point[dp_id], devide_point[dp_id+1]),]))
                    
                else:
                    target_date_modified.append(deepcopy(tmp_target))
                
            else:
                target_date_modified.append(deepcopy(tmp_target))
    
        # 補正結果をpd.Series化
        target_date_modified = pd.concat(target_date_modified, axis=0)
        target_date_modified.index = target_date.index
        
        # 補正結果を置き換え
        df.loc[:,date_list[date_id]] = target_date_modified
        
    return df




# データ読み込み
track = "C"
file_name = f"track_{track}.csv"
df_track = pd.read_csv(f"input/{file_name}")
df_track.head()


columns = "高低左"
df_irregularity = []
date_list = list(df_track.loc[:,"date"].unique())
milage_list = list(df_track.loc[:,"キロ程"].unique())

for milage_id in range(len(milage_list)):
    milage_list[milage_id] = "m" + str(milage_list[milage_id])


print("行：キロ程，列：日付idのデータ作成")
time.sleep(0.5)
for date in tqdm(date_list):
    # 対象キロ程のデータを取得
    tmp_df = df_track[df_track.loc[:,"date"]==date].loc[:,[columns]]
    
    # インデックスを初期化
    tmp_df = tmp_df.reset_index(drop=True)
    
    # リネームしてアペンド
    df_irregularity.append(tmp_df.rename(columns={columns: f"{date}"}))

df_irregularity = pd.concat(df_irregularity, axis=1)
df_irregularity.head
df_irregularity.tail
df_irregularity.shape


# 位相補正
df_irregularity_phase_modified = phaseModify(df=deepcopy(df_irregularity) ,n_shift=20, n_sub_data=500)

# 列がキロ程になるよう転置
df_irregularity_phase_modified = df_irregularity_phase_modified.T

# 列名変更
df_irregularity_phase_modified.columns = milage_list

# 保存
df_irregularity_phase_modified.to_csv(f"input/irregularity_{track}_phase_modified.csv",index=False,header=True)



