#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

 後処理用関数 
 
"""

import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import time


# 初期値のみで逐次予測を実行
def predOnlyStart(start_raw, t_pred):
    pred_raw_list = []
    for i in range(t_pred):
        pred_raw_list.append(start_raw)
    return pred_raw_list


#def diagnosePredResult(df_pred, df_train, tol_abnormal_max_min = 2.5, tol_abnormal_upper = 25, tol_abnormal_lower = -25):
#    # 検査結果保存場所
#    diagnosis_result = {}
#    
#    # 最大値-最小値が訓練データの最大値-最小値のtol_abnormal_max_min倍以上のもの
#    train_max_min = df_train.max() - df_train.min()
#    pred_max_min = df_pred.max() - df_pred.min()
#    abnormal_max_min  = pred_max_min > train_max_min*tol_abnormal_max_min
#    diagnosis_result["abnormal_max_min"] = abnormal_max_min
#    
#    # 最大値がtol_abnormal_upperを越すもの
#    abnormal_upper = df_pred.max() > tol_abnormal_upper
#    diagnosis_result["abnormal_upper"] = abnormal_upper
#    
#    # 最小値がtol_abnormal_lowerを下回るもの
#    abnormal_lower = df_pred.min() < tol_abnormal_lower
#    diagnosis_result["abnormal_lower"] = abnormal_lower
#    
#    # 全ての検査項目の和集合を出力
#    abnormal_total = abnormal_max_min | abnormal_upper | abnormal_lower
#    
#    return abnormal_total, diagnosis_result

# (変更)
def diagnosePredResult(df_pred, train_max_min, tol_abnormal_max_min = 2.5, tol_abnormal_upper = 25, tol_abnormal_lower = -25):
    # 検査結果保存場所
    diagnosis_result = {}
    
    # 最大値-最小値が訓練データの最大値-最小値のtol_abnormal_max_min倍以上のもの
    pred_max_min = df_pred.max() - df_pred.min()
    abnormal_max_min  = pred_max_min > train_max_min*tol_abnormal_max_min
    diagnosis_result["abnormal_max_min"] = abnormal_max_min
    
    # 最大値がtol_abnormal_upperを越すもの
    abnormal_upper = df_pred.max() > tol_abnormal_upper
    diagnosis_result["abnormal_upper"] = abnormal_upper
    
    # 最小値がtol_abnormal_lowerを下回るもの
    abnormal_lower = df_pred.min() < tol_abnormal_lower
    diagnosis_result["abnormal_lower"] = abnormal_lower
    
    # 全ての検査項目の和集合を出力
    abnormal_total = abnormal_max_min | abnormal_upper | abnormal_lower
    
    return abnormal_total, diagnosis_result


#def postTreat(df_pred_raw, abnormal_total, start_raw_dict, t_pred, method):
#    # 後処理前のデータをコピー
#    df_pred_raw_post = deepcopy(df_pred_raw)
#    
#    # キロ程リスト取得
#    milage_list = list(df_pred_raw_post.columns)
#        
#    for milage_id in tqdm(range(len(milage_list))):
#        milage = milage_list[milage_id]
#        target_start = start_raw_dict[milage]
#        
#        if abnormal_total[milage]:
#            
#            if method=="average":
#                # 前後共ない場合，初期値のみで予測結果を修正
#                modified_pred = predOnlyStart(start_raw_dict[milage], t_pred)
#                
#            else:
#                # となりのキロ程を取得
#                if milage_id==0:
#                    next_milage_list = [milage_list[milage_id+1]]
#                elif milage_id==(len(milage_list)-1):
#                    next_milage_list = [milage_list[milage_id-1]]
#                else:
#                    next_milage_list = [milage_list[milage_id-1], milage_list[milage_id+1]]
#                
#                # となりのキロ程にover_tol==Falseがあるかチェック
#                tmp_not_over_tol = abnormal_total[next_milage_list]==False
#                donor_milage_list = list(tmp_not_over_tol[tmp_not_over_tol].index)
#                
#                if len(donor_milage_list)==2:
#                    # 前後いずれもある場合，前後の予測結果の平均値+(ターゲットの初期値-前後の初期値の平均値)で修正
#                    front_pred = deepcopy(df_pred_raw_post.loc[:,donor_milage_list[0]])
#                    front_start = start_raw_dict[donor_milage_list[0]]
#                    
#                    back_pred = deepcopy(df_pred_raw_post.loc[:,donor_milage_list[1]])
#                    back_start = start_raw_dict[donor_milage_list[1]]
#                    
#                    donor_pred = (front_pred + back_pred) / 2.0
#                    donor_start = (front_start + back_start) / 2.0
#                    
#                    modified_pred = donor_pred + (target_start - donor_start)
#                    
#                    
#                elif len(donor_milage_list)==1:
#                    # 片一方の場合，ドナーの予測結果+(ターゲットの初期値-ドナーの初期値)で修正
#                    donor_pred = deepcopy(df_pred_raw_post.loc[:,donor_milage_list[0]])
#                    donor_start = start_raw_dict[donor_milage_list[0]]
#                    
#                    modified_pred = donor_pred + (target_start - donor_start)
#                
#                else:
#                    # 前後共ない場合，初期値のみで予測結果を修正
#                    modified_pred = predOnlyStart(start_raw_dict[milage], t_pred)
#            
#            # 予測結果を修正
#            modified_pred = pd.Series(modified_pred)
#            modified_pred.index = df_pred_raw_post[milage].index
#            df_pred_raw_post[milage] = modified_pred
#            
#    return df_pred_raw_post
#
#    

# (変更)
def postTreat(df_pred_raw, abnormal_total, init_raw_dict, t_pred):
    # 後処理前のデータをコピー
    df_pred_raw_post = deepcopy(df_pred_raw)
    
    # キロ程リスト取得
    milage_list = list(df_pred_raw_post.columns)
        
    for milage_id in tqdm(range(len(milage_list))):
        milage = milage_list[milage_id]
        target_start = init_raw_dict[milage]
        
        if abnormal_total[milage]:
           
            # となりのキロ程を取得
            if milage_id==0:
                next_milage_list = [milage_list[milage_id+1]]
            elif milage_id==(len(milage_list)-1):
                next_milage_list = [milage_list[milage_id-1]]
            else:
                next_milage_list = [milage_list[milage_id-1], milage_list[milage_id+1]]
            
            # となりのキロ程にover_tol==Falseがあるかチェック
            tmp_not_over_tol = abnormal_total[next_milage_list]==False
            donor_milage_list = list(tmp_not_over_tol[tmp_not_over_tol].index)
            
            if len(donor_milage_list)==2:
                # 前後いずれもある場合，前後の予測結果の平均値+(ターゲットの初期値-前後の初期値の平均値)で修正
                front_pred = deepcopy(df_pred_raw_post.loc[:,donor_milage_list[0]])
                front_start = init_raw_dict[donor_milage_list[0]]
                
                back_pred = deepcopy(df_pred_raw_post.loc[:,donor_milage_list[1]])
                back_start = init_raw_dict[donor_milage_list[1]]
                
                donor_pred = (front_pred + back_pred) / 2.0
                donor_start = (front_start + back_start) / 2.0
                
                modified_pred = donor_pred + (target_start - donor_start)
                
                
            elif len(donor_milage_list)==1:
                # 片一方の場合，ドナーの予測結果+(ターゲットの初期値-ドナーの初期値)で修正
                donor_pred = deepcopy(df_pred_raw_post.loc[:,donor_milage_list[0]])
                donor_start = init_raw_dict[donor_milage_list[0]]
                
                modified_pred = donor_pred + (target_start - donor_start)
            
            else:
                # 前後共ない場合，初期値のみで予測結果を修正
                modified_pred = predOnlyStart(init_raw_dict[milage], t_pred)
            
            # 予測結果を修正
            modified_pred = pd.Series(modified_pred)
            modified_pred.index = df_pred_raw_post[milage].index
            df_pred_raw_post[milage] = modified_pred
            
    return df_pred_raw_post

    
def makeSubmitFile():
    print("\n・予測結果を提出フォーマットに変換し出力 ===============================")
    time.sleep(0.5)
    
    # index_master読み込み
    df_index_master = pd.read_csv(f"input/index_master.csv")
    
    # 日付リスト読み込み
    df_date = pd.read_csv(f"input/date.csv",index_col=0)
    
    # 結果リストを初期化
    result_list = []
    
    for track in tqdm(["A","B","C","D"]):
        # 予測結果データ読み込み
        df_tmp_pred = pd.read_csv(f"output/Predicting/pred_track_{track}.csv")  
        
        # もし欠損カラムがあれば，直前のキロ程で補完  
        missing_milage_list = list(df_tmp_pred.columns[df_tmp_pred.isna().any(axis=0)])
        for missing_milage in missing_milage_list:
            missing_milage_id = int(missing_milage.split("m")[1])
            df_tmp_pred[missing_milage] = deepcopy(df_tmp_pred.loc[:,f"m{missing_milage_id-1}"])
    
        # indexを日付に変更
        df_tmp_pred.index = df_date["date"]
        
        # 転置
        df_tmp_pred = df_tmp_pred.T
            
        # 日付-キロ程の順で1列にデータをまとめる
        tmp_list = []
        for date in list(df_date["date"]):
            tmp_list.append(df_tmp_pred[date])
        tmp_list = pd.concat(tmp_list, axis=0)
        
        # 結果リストにアペンド
        result_list.append(tmp_list)
        
    # 結果リストをDataFrame化
    result_list = pd.concat(result_list, axis=0)   

    # 結果リストのインデックスを提出フォーマットのインデックスに変更
    #result_list.index = df_index_master.index
    
    # 保存
    result_list.to_csv(f"output/Predicting/result_for_submit.csv", index=True, header=False)

    