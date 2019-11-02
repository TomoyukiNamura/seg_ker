# データベースコマンド
show databases;            # データベース一覧
-- create database test;      # データベース作成
-- drop database test;        # データベース削除
use test;                  # データベース切り替え
select database();         # 接続中のデータベース表示

# テーブル一覧
show tables;

# テーブル作成
CREATE TABLE features (
 id INT NOT NULL AUTO_INCREMENT,
 feature JSON NOT NULL,
 PRIMARY KEY (id)
);
-- drop table features;

# カラム一覧
show columns from features;

# jsonファイル読み込み
LOAD DATA local infile '/Users/tomoyuki/Desktop/features.json' INTO TABLE features (feature);

## データ情報
###  データセレクト
select * from features;

## レコード数
SELECT COUNT(*) FROM features;

## 条件検索
### 下記構造のjsonを想定
-- { 
-- 	"type": "Feature", 
-- 	"properties": { 
-- 		...
-- 		"ODD_EVEN": "O" 
-- 	}, 
-- 	...
-- }
SELECT * FROM features WHERE JSON_CONTAINS(feature, '"O"', '$.properties.ODD_EVEN') ;
SELECT COUNT(*) FROM features WHERE JSON_CONTAINS(feature, '"O"', '$.properties.ODD_EVEN') ;



