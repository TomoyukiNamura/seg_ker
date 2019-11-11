#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Learning

"""


import os
os.chdir("/Users/tomoyuki/python_workspace/new_train_conte")

import time
import configparser

from func import learn_func
            

def main():
    for track in ["A","B","C","D"]:
        
        print(f"\ntrack{track} ===============================")
        time.sleep(0.5)
        
        ## 初期処理 
        ## 設定ファイル読み込み
        config = configparser.ConfigParser()
        config.read(f"conf/main.ini", 'UTF-8')
        ridge_alpha = config.getfloat('learn', 'ridge_alpha')
        
    
        # (追加)データ読み込み
        train_data_dict, n_orgdata_dict = learn_func.readData(f"output/Preprocessing/track_{track}")
            
        ## 学習
        learn_func.train(train_data_dict, ridge_alpha, n_orgdata_dict, f"output/Learning/track_{track}")
            
        
        
if __name__ == "__main__":
    main()
    