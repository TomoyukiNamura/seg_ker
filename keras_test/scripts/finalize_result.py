#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir("/Users/tomoyuki/Desktop/keras_test")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

# ファイナライズ対象のフォルダ名
finalize_folder_name = "20190104"


## 
# データ読み込み
df_index_master = pd.read_csv(f"input/index_master.csv")
df_index_master.head()

df_date = pd.read_csv(f"input/date.csv",index_col=0)
df_date.head()


result_list = []
missing_dict = {}

for alphabet in ["A","B","C","D"]:
    # 予測結果データ読み込み
    df_tmp_pred = pd.read_csv(f"output/{finalize_folder_name}/pred_track_{alphabet}.csv")  
    
    # 欠損カラムは直前のキロ程で補完  
    missing_milage_list = list(df_tmp_pred.columns[df_tmp_pred.isna().any(axis=0)])
    
    missing_dict[alphabet] = missing_milage_list
    for missing_milage in missing_milage_list:
        missing_milage_id = int(missing_milage.split("m")[1])
        df_tmp_pred[missing_milage] = deepcopy(df_tmp_pred.loc[:,f"m{missing_milage_id-1}"])
        

    # indexを日付に変更
    df_tmp_pred.index = df_date["date"]
    
    # 転置
    df_tmp_pred = df_tmp_pred.T
        
    # 
    tmp_list = []
    for date in tqdm(list(df_date["date"])):
        tmp_list.append(df_tmp_pred[date])
    tmp_list = pd.concat(tmp_list, axis=0)
    
    # 
    result_list.append(tmp_list)
        
result_list = pd.concat(result_list, axis=0)          
result_list.index =df_index_master.index


## 大きくハズレた値がないか調査
tmp_bool = np.abs(result_list) > 30
n_upper30 = result_list[tmp_bool].shape[0]
plt.plot(np.array(result_list));plt.grid();plt.title(f"n_upper30: {n_upper30}")
plt.savefig(f"output/{finalize_folder_name}/plot_{finalize_folder_name}.jpg");plt.show()


# finalizeファイル作成
result_list.to_csv(f"output/{finalize_folder_name}/finalize_{finalize_folder_name}.csv", index=True, header=False)

# フォルダ名を_finalizedに変更
is_success = False
folder_id = 1
while(is_success==False):
    candidate_folder_name = f"output/{finalize_folder_name}_finalized_{folder_id}"
    
    if os.path.exists(candidate_folder_name)==False:
        os.rename(f"output/{finalize_folder_name}", candidate_folder_name)
        is_success = True
    
    else:
        folder_id += 1


