#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

APP_NAME = "/Users/tomoyuki/python_workspace/takeda"
os.chdir(APP_NAME)
sys.path.append(os.path.join(APP_NAME, "src"))

import utils

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import time
import shutil
import configparser

# 設定
OUTPUT_FOLDER_NAME = "valid2"
RANDOM_SEED        = 0
TRAIN_SIZE         = 0.8


# 出力フォルダ作成
utils.make_new_folder(f"data/{OUTPUT_FOLDER_NAME}")

# データ読み込み
df_org = pd.read_csv(f"data/train.csv")


# 訓練データのサンプルIDを取得
train_ids = df_org.sample(n=round(df_org.shape[0]*TRAIN_SIZE), random_state=RANDOM_SEED).index.sort_values()

# 訓練データ出力
df_org.loc[train_ids,:].to_csv(f"data/{OUTPUT_FOLDER_NAME}/train.csv",index=False, header=True)

# 評価データ出力
df_org.drop(train_ids).to_csv(f"data/{OUTPUT_FOLDER_NAME}/test.csv",index=False, header=True)


## 各特徴量と目的変数とのマハラノビス距離計算
#X_train = df_org.loc[train_ids,:].iloc[:, 2:df_org.shape[1]]
#y_train = df_org.loc[train_ids,:].iloc[:, [1]]
#df_mah_dis = utils.make_df_mah_dis(y_train, X_train)
#df_mah_dis.to_csv(f"data/{OUTPUT_FOLDER_NAME}/train_mah_dis.csv",index=False, header=True)