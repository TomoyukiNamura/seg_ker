#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
os.chdir("/Users/tomoyuki/python_workspace/NLP/VNC")
#import MeCab
import sqlite3
from contextlib import closing
import pandas as pd


df_dict = pd.read_csv("df_dict_clen1.csv")



# dbに辞書を登録
dbname = "EDICT_cleaned.sqlite3"

# データベースに接続するには，sqlite3.connect()メソッドを使用する->Connectionオブジェクトが作成される
conn = sqlite3.connect(dbname)

# テーブル作成
columns_sql = "(id integer primary key autoincrement"
for i in range(1,df_dict.shape[1]):
    column = df_dict.columns[i]
    columns_sql = columns_sql + f", {column} text"
columns_sql = columns_sql + ")"

sql = f"create table edict_cleaned {columns_sql}"
cur = conn.cursor()
cur.execute(sql)
cur.close()


## テーブル削除
#cur = conn.cursor()
#cur.execute("drop table edict_cleaned")
#cur.close()




# データベースに辞書を登録
columns_sql = f"({df_dict.columns[1]}"
questions_sql = "(?"

for i in range(2,df_dict.shape[1]):
    column = df_dict.columns[i]
    columns_sql = columns_sql + f", {column}"
    questions_sql = questions_sql + ", ?"
    
columns_sql = columns_sql + ")"
questions_sql = questions_sql + ")"  


cur = conn.cursor()
sql = f"insert into edict_cleaned {columns_sql} values {questions_sql}"

for row_id in range(df_dict.shape[0]):
    # インサートする要素を作成
    dic = []
    for col_id in range(1,df_dict.shape[1]):
        try:
            dic.append(df_dict.iloc[row_id,col_id])
        except:
            dic.append("")
    dic = tuple(dic)
    
    # インサート
    cur.execute(sql, dic)

conn.commit()
cur.close()








# 登録できているか確認
# itemsテーブルのカラム名取得
cur = conn.cursor()
select_sql = "pragma table_info(edict_cleaned);" 
for row in cur.execute(select_sql):
    print_txt = ""
    for i in range(len(row)):
        print_txt = f"{print_txt}{row[i]} | "
    print(print_txt)
cur.close()


select_sql = f"select * from edict_cleaned" 
cur = conn.cursor()
for row in cur.execute(select_sql):
    print_txt = ""
    for i in range(len(row)):
        if row[i]!=None:
            print_txt = f"{print_txt}{row[i]} | "
    
    print(print_txt)
cur.close()    

