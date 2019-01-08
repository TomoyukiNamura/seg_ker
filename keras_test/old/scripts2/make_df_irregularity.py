#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
os.chdir("/Users/tomoyuki/Desktop/keras_test")

import pandas as pd
from tqdm import tqdm
import time


track_list = ["A", "B", "C", "D"]


for track in track_list:
    # データ読み込み
    file_name = f"track_{track}.csv"
    df_track = pd.read_csv(f"input/{file_name}")
    df_track.head()
    
    
    columns = "高低左"
    df_irregularity = []
    milage_list = list(df_track.loc[:,"キロ程"].unique())
    
    print("行：日付id, 列：キロ程　のデータ作成")
    time.sleep(0.5)
    for milage in tqdm(milage_list):
        # 対象キロ程のデータを取得
        tmp_df = df_track[df_track.loc[:,"キロ程"]==milage].loc[:,[columns]]
        
        # インデックスを初期化
        tmp_df = tmp_df.reset_index(drop=True)
        
        # リネームしてアペンド
        df_irregularity.append(tmp_df.rename(columns={columns: f"m{milage}"}))
    
    df_irregularity = pd.concat(df_irregularity, axis=1)
    
    # 保存
    df_irregularity.to_csv(f"input/irregularity_{track}.csv",index=False,header=True)

    time.sleep(5)
