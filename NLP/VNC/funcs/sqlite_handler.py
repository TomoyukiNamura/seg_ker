#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# テーブル作成
def createTable(conn, table_name, columns_sql):
    sql = f"create table {table_name} {columns_sql}"
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()


# テーブルのカラム名取得
def selectColumns(conn, table_name):
    cur = conn.cursor()
    select_sql = f"pragma table_info({table_name})" 
    for row in cur.execute(select_sql):
        print_txt = ""
        for i in range(len(row)):
            print_txt = f"{print_txt}{row[i]} | "
        print(print_txt)
    cur.close()


# テーブルのレコード取得
def selectRecords(conn, table_name):
    select_sql = f"select * from {table_name}" 
    cur = conn.cursor()
    for row in cur.execute(select_sql):
        print_txt = ""
        for i in range(len(row)):
            if row[i]!=None:
                print_txt = f"{print_txt}{row[i]} | "
        
        print(print_txt)
    cur.close()    

