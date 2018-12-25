#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 17:29:36 2018

@author: tomoyuki
"""




#### データ調査 =======================================
### データ数は全データに対して何割か？
#max_n_data = df_irregularity.shape[0]
#n_data_list = [round(0.1*max_n_data), round(0.6*max_n_data)]
#
#r_data_category_dict = {}
#
#for milage in list(df_irregularity.columns):
#    # データ数取得
#    tmp_n_data = df_irregularity[milage].dropna().shape[0]
#    
#    # 
#    if tmp_n_data >= n_data_list[1]:
#        r_data_category_dict[milage] = "a"
#        
#    elif tmp_n_data < n_data_list[1] and tmp_n_data >= n_data_list[0]:
#        r_data_category_dict[milage] = "b"
#        
#    elif tmp_n_data < n_data_list[0]:
#        r_data_category_dict[milage] = "c"
#        
#    else:
#        r_data_category_dict[milage] = np.nan
#
## 結果
#print(df_irregularity.shape[1])
#print(len(r_data_category_dict))
#
#r_data_category_values = np.array(list(r_data_category_dict.values()))
#np.unique(r_data_category_values, return_counts=True)
#
#
#milage = "m30597"
#r_data_category_dict[milage]
#df_irregularity[milage].dropna().shape
#
#
### 前後n_nextキロ程農のうち，r_data_category="a"が含まれているか？
#n_next = 3


## 初期値データ取得 ====================================================================
start_dict = {}
for key in list(org_dict.keys()):
    start_dict[key] = deepcopy(org_dict[key].iloc[start_date_id_list,:])



