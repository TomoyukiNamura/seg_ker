#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Predicting

"""


import os
os.chdir("/Users/tomoyuki/python_workspace/new_train_conte")

from copy import deepcopy
import time
import configparser

from func import predict_func


def main():
    for track in ["A","B","C","D"]:
        
        print(f"\ntrack{track} ===============================")
        time.sleep(0.5)
        
        ## 初期処理 
        # 設定ファイル読み込み
        config = configparser.ConfigParser()
        config.read(f"conf/main.ini", 'UTF-8')
        tol_abnormal_max_min = config.getfloat('predict', 'tol_abnormal_max_min')
        tol_abnormal_upper = config.getfloat('predict', 'tol_abnormal_upper')
        tol_abnormal_lower = config.getfloat('predict', 'tol_abnormal_lower')
            
        
        ## (追加)データ読み込み
        model_dict, init_raw_dict, init_diff_dict, train_max_min = predict_func.readData(f"output/Preprocessing/track_{track}", f"output/Learning/track_{track}")
            
        
        ## 逐次予測
        df_pred_raw = predict_func.predWithARIMA(model_dict=model_dict, init_raw_dict=init_raw_dict, init_diff_dict=init_diff_dict, t_pred=91)
        
        
        # (変更)予測結果を検査し，異常値を取得
        abnormal_total, _ = predict_func.diagnosePredResult(df_pred=deepcopy(df_pred_raw), train_max_min=train_max_min, tol_abnormal_max_min = tol_abnormal_max_min, tol_abnormal_upper = tol_abnormal_upper, tol_abnormal_lower = tol_abnormal_lower)
                
        # 異常値の除去
        df_pred_raw = predict_func.postTreat(df_pred_raw=df_pred_raw, abnormal_total=abnormal_total, init_raw_dict=init_raw_dict, t_pred=91)
        
        # 予測対象範囲外の除去
        df_pred_raw = df_pred_raw.iloc[range(91),:]
        
        # 予測結果を保存
        df_pred_raw.to_csv(f"output/Predicting/pred_track_{track}.csv",index=False)
        time.sleep(0.5)
        
    
    ## 予測結果を提出フォーマットに変換し出力
    predict_func.makeSubmitFile()
        
        
if __name__ == "__main__":
    main()
    