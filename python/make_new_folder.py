#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## パッケージインポート
import os
import shutil
import numpy as np

## ディレクトリ変更
os.chdir('myfoldername')
current_pass = os.getcwd()

# フォルダ作成関数
def makeNewFolder(folder_name):
    if os.path.exists(folder_name)==False:
        os.mkdir(folder_name)
        


# carlaフォルダ内のフォルダをnear_miss,no_near_missごとに勘定
carla_folder_name = "carla"
carla_folder_list = os.listdir(path=os.path.join(current_pass,carla_folder_name))


# near_miss,no_near_missのラベル付け
bool_near_miss = []
bool_no_near_miss = []

for carla_folder in carla_folder_list:
    
    # 名前にnearmissが入っているかチェック
    if "nearmiss" in carla_folder:
        
        # 名前にnonearmissが入っているかチェック
        if "nonearmiss" in carla_folder:
            # carla_folderはno_near_miss
            bool_near_miss.append(False)
            bool_no_near_miss.append(True)
            
        else:
            # carla_folderはnear_miss
            bool_near_miss.append(True)
            bool_no_near_miss.append(False)
        
    else:
        # carla_folderはどちらでもない
        bool_near_miss.append(False)
        bool_no_near_miss.append(False)




## 作成対象フォルダ名定義
first_folder_name_list = ["test_image"]
second_folder_name_list = ["near_miss","no_near_miss"]
third_folder_name_list = []


# 大元のフォルダ作成
for first_folder_name in first_folder_name_list:
    output_pass = first_folder_name
    makeNewFolder(output_pass)
    
    # near_miss,no_near_missフォルダ作成
    for second_folder_name in second_folder_name_list:
        output_pass = os.path.join(first_folder_name,second_folder_name)
        makeNewFolder(output_pass)
        
        # carlaからフォルダをコピー
        if second_folder_name=="near_miss":
            copy_org_list = list(np.array(carla_folder_list)[bool_near_miss])
            
        elif second_folder_name=="no_near_miss":
            copy_org_list = list(np.array(carla_folder_list)[bool_no_near_miss])
            
        for copy_org in copy_org_list:
            copy_org_pass = os.path.join(carla_folder_name,copy_org)
            copy_dist_pass = os.path.join(output_pass,copy_org)
            shutil.copytree(copy_org_pass,copy_dist_pass)
            
            # imgフォルダ作成
            makeNewFolder(os.path.join(copy_dist_pass,"img"))

            # .pngファイルをimgフォルダへ移動
            for file in os.listdir(copy_dist_pass):
                if ".png" in file:
                    shutil.move(os.path.join(copy_dist_pass,file), os.path.join(copy_dist_pass,"img",file))
