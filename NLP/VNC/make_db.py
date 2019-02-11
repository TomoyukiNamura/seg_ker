#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir("/Users/tomoyuki/python_workspace/NLP/VNC")
#import MeCab
import sqlite3
from contextlib import closing


dbname = "EDICT.sqlite3"


# データベースに接続するには，sqlite3.connect()メソッドを使用する->Connectionオブジェクトが作成される
conn = sqlite3.connect(dbname)

# itemsテーブルのカラム名取得
cur = conn.cursor()
select_sql = "pragma table_info(edict);" 
for row in cur.execute(select_sql):
    print_txt = ""
    for i in range(len(row)):
        print_txt = f"{print_txt}{row[i]} | "
    print(print_txt)
cur.close()


# 1行ずつ読み込む(SQL文を実行するには，Cursorオブジェクトのexecute()メソッドを使用する．)
select_sql = "select * from edict" 
cur = conn.cursor()
for row in cur.execute(select_sql):
    print_txt = ""
    for i in range(len(row)):
        print_txt = f"{print_txt}{row[i]} | "
    print(print_txt)
cur.close()






## EDICTデータベース作成

import os
os.chdir("/Users/tomoyuki/python_workspace/NLP/VNC")
#import MeCab
import sqlite3
from contextlib import closing



# txtファイル読み込み，リストに保存

## テスト用 ========
#max_nline = 100
## テスト用 ========

fname = "edict2u.txt"

row_list = []
len_row_list = []

with open(fname, "r", encoding="utf-8") as f:
#    print(f.read())
    i = 0
    for line in f:
        # 一行取得
        tmp_line = line.split("/")
        
        # 末尾の改行コードを除外
        del tmp_line[len(tmp_line)-2 : len(tmp_line)]
        
        # row_listに保存
        row_list.append(tmp_line)
        
        # tmp_lineの長さをlen_row_listに保存
        len_row_list.append(len(tmp_line))
    
#        # テスト用 ========
#        i+=1
#        if i==max_nline:
#            break
#        # テスト用 ========

# 結果
print(row_list)
print(len_row_list)


# dbに辞書を登録
dbname = "EDICT.sqlite3"

# データベースに接続するには，sqlite3.connect()メソッドを使用する->Connectionオブジェクトが作成される
conn = sqlite3.connect(dbname)


# テーブル作成
columns_sql = "(id integer primary key autoincrement, jpn text"
for i in range(max(len_row_list)):
    column = f"eng{i+1}"
    columns_sql = columns_sql + f", {column} text"
columns_sql = columns_sql + ")"
    
sql = f"create table edict {columns_sql}"
cur = conn.cursor()
cur.execute(sql)
cur.close()


## テーブル削除
#cur = conn.cursor()
#cur.execute("drop table edict")
#cur.close()




# データベースに辞書を登録
columns_sql = "(jpn"
questions_sql = "(?"

for i in range(max(len_row_list)):
    column = f"eng{i+1}"
    columns_sql = columns_sql + f", {column}"
    questions_sql = questions_sql + ", ?"
    
columns_sql = columns_sql + ")"
questions_sql = questions_sql + ")"  
    
    
cur = conn.cursor()
sql = f"insert into edict {columns_sql} values {questions_sql}"

for row_id in range(len(row_list)):
    # インサートする要素を作成
    dic = []
    for i in range(max(len_row_list)+1):
        try:
            dic.append(row_list[row_id][i])
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
select_sql = "pragma table_info(edict);" 
for row in cur.execute(select_sql):
    print_txt = ""
    for i in range(len(row)):
        print_txt = f"{print_txt}{row[i]} | "
    print(print_txt)
cur.close()


select_sql = "select * from edict" 
cur = conn.cursor()
for row in cur.execute(select_sql):
    print_txt = ""
    for i in range(len(row)):
        print_txt = f"{print_txt}{row[i]} | "
    print(print_txt)
cur.close()





