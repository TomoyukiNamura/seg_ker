#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

出力結果があっているかチェック

"""

import os
os.chdir("/Users/tomoyuki/python_workspace/new_train_conte")

import pandas as pd

true_path = "output/Predicting/true_test"
target_path = "output/Predicting"


tmp_list = []

for track in ["A","B","C","D"]:
    # 読み込み
    true = pd.read_csv(f"{true_path}/pred_track_{track}.csv")
    target = pd.read_csv(f"{target_path}/pred_track_{track}.csv")
    
    # 照合
    tmp_list.append((true == target).all(axis=0).all(axis=0))

print(f"result: {all(tmp_list)}")
