#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
os.chdir("/Users/tomoyuki/python_workspace/NLP/VNC")
import MeCab
import sqlite3
from contextlib import closing
import pandas as pd
import pandas.io.sql as psql
import numpy as np
from copy import deepcopy
import re
import jaconv
import traceback
from tqdm import tqdm

from funcs import char_handler

char_handler.removeBracket
char_handler.checkCharacterType
char_handler.removeCharInString

## DBに接続
dbname = "EDICT.sqlite3"
conn = sqlite3.connect(dbname)

selectColumns(conn, "code_1332")
selectRecords(conn, "code_1332")

psql.read_sql("SELECT * FROM articles;", con)

# itemsテーブルのカラム名取得
column_list = []
cur = conn.cursor()
select_sql = "pragma table_info(edict);" 
for row in cur.execute(select_sql):
    print_txt = ""
    for i in range(len(row)):
        print_txt = f"{print_txt}{row[i]} | "
    print(print_txt)
    column_list.append(row[1])
cur.close()



## 一部分をpandas化
df_dict = pd.DataFrame(data=None, columns=column_list)

#id_list = list(range(0,180000,500))

select_sql = "select * from edict" 
cur = conn.cursor()
i=0

for row in tqdm(list(cur.execute(select_sql))):
#    if i in id_list:
    # DataFarmeに値(行)を追加していく
    tmp_se = pd.Series(row, index=df_dict.columns )
    df_dict = df_dict.append( tmp_se, ignore_index=True )
        
    i+=1
    
cur.close()


# 保存
df_dict = df_dict.drop(0, axis=0)
df_dict.to_csv("df_dict_all.csv", encoding="utf-8",index=False)





##        
##
#string="くぁwせdrfヲヲヲエええ?$#ーーうえ"
#char_handler.removeCharInString(string ,ctype="hira")
##
##
### カタカナ2ひらがな
#char_handler.removeCharInString(jaconv.kata2hira(string),ctype="hira")
#char_handler.removeCharInString(jaconv.hira2kata(string),ctype="kata")


## df_dict読み込み
df_dict = pd.read_csv("df_dict.csv")


## 1.日本語文字が複数のもの(;)はレコードを分ける
# カラム名初期化
columns = list(df_dict.columns)
columns.insert(2, "jpn_hira")
columns.insert(3, "jpn_kata")

# df初期化
df_dict_clen1 = pd.DataFrame(data=None, columns=columns)
df_dict_clen1_remove = pd.DataFrame(data=None, columns=columns[0:3])



row_id = 182



for row_id in range(df_dict.shape[0]):
    # 一行取得
    tmp_row = deepcopy(df_dict.iloc[row_id,:])
    
    # 英単語中の()文字を削除
    remove_dict = {}
    tmp_row_index = list(tmp_row.index)
    for i in range(2,tmp_row.shape[0]):
        tmp_str,remove_dict[tmp_row_index[i]] = char_handler.removeBracket(string=tmp_row[tmp_row_index[i]])
       
        # 英単語
        if type(tmp_str) is str:
            tmp_str_split = tmp_str.split(" ")
            while "" in tmp_str_split :
                tmp_str_split.remove("")
            tmp_row[tmp_row_index[i]] = " ".join(tmp_str_split)
            
        else:
            tmp_row[tmp_row_index[i]] = tmp_str
    
    
    
    # 日本語を取得
    jpn = tmp_row["jpn"]
    
    # 漢字と読み仮名に分割(空欄で分割)
    jpn_org = jpn.split(" ")[0]
    jpn_readings = jpn.split(" ")[1]
    
    
    ## 漢字をクレンジング
    # ()文字を削除
    jpn_org, remove_org = char_handler.removeBracket(jpn_org)
    
    
    ## 読み仮名をクレンジング
    # 外側の[]削除
    jpn_readings = jpn_readings[1:-1]
    
    # ()文字を削除
    jpn_readings, remove_readings = char_handler.removeBracket(string=jpn_readings)
    
    # ";"で分割
    jpn_readings_list = []
    for part_jpn_readings in jpn_readings.split(";"):
        # カタカナ2ひらがな
        part_jpn_readings = jaconv.kata2hira(part_jpn_readings)
        
        # ひらがな以外の文字を削除
        part_jpn_readings = char_handler.removeCharInString(part_jpn_readings ,ctype="hira")
        
        # すでに対象文字(part_jpn_readings)がnew_jpn_readingsに含まれていない場合，追加
        if part_jpn_readings not in jpn_readings_list:
            jpn_readings_list.append(part_jpn_readings)
            
    
    
    ## 漢字リスト(jpn_org.split(";"))x読み仮名リスト(jpn_readings_list)でレコード追加
    for part_jpn in jpn_org.split(";"):
        
        for part_jpn_readings in jpn_readings_list:
            
            # 読み仮名(part_jpn_readings)がない場合は，part_jpnをひらがなに直し，挿入
            if part_jpn_readings=="":
                part_jpn_readings = jaconv.kata2hira(part_jpn)
                        
            # 追加するレコードを初期化
            new_row = deepcopy(tmp_row)
            
            # jpnをpart_jpnに変換
            new_row["jpn"] = part_jpn
            
            # 読み仮名(ひらがな)を追加
            new_row["jpn_hira"] = part_jpn_readings
            
            # 読み仮名(カタカナ)を追加
            new_row["jpn_kata"] = jaconv.hira2kata(part_jpn_readings)
            
            # 追加
            df_dict_clen1 = df_dict_clen1.append(new_row, ignore_index=True )
            
            # 除外した文字列を格納
            remove_dict["id"] = new_row["id"]
            remove_dict["jpn"] = remove_org
            remove_dict["jpn_hira"] = remove_readings
                    
            df_dict_clen1_remove = df_dict_clen1_remove.append(pd.Series(remove_dict), ignore_index=True )

## 保存
df_dict_clen1.to_csv("df_dict_clen1.csv", encoding="utf-8",index=False)
df_dict_clen1_remove.to_csv("df_dict_clen1_remove.csv", encoding="utf-8",index=False)





