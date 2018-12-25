#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

# ファイナライズ対象のフォルダ名
finalize_folder_name = "20181226"


## 
# データ読み込み
df_index_master = pd.read_csv(f"input/index_master.csv")
df_index_master.head()

df_date = pd.read_csv(f"input/date.csv",index_col=0)
df_date.head()

#df_sumple = pd.read_csv(f"input/sample_submit.csv")
#df_sumple.shape

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
        
        #df_tmp_pred[missing_milage] = 25

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

#result_list.shape
#df_index_master.shape

# finalizeファイル作成
result_list.to_csv(f"output/{finalize_folder_name}/finalize_{finalize_folder_name}.csv", index=True, header=False)

# フォルダ名を_finalizedに変更
is_success = False
finalize_folder_name_renamed = deepcopy(finalize_folder_name)
while(is_success==False):
    if os.path.exists(f"output/{finalize_folder_name_renamed}_finalized")==False:
        os.rename(f"output/{finalize_folder_name}", f"output/{finalize_folder_name_renamed}_finalized")
        is_success = True
    
    else:
        finalize_folder_name_renamed = f"{finalize_folder_name_renamed}_a"


for track in list(missing_dict.keys()):
    print(f"{track}: {len(missing_dict[track])}")
    
