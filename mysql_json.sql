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
select * from features limit 3;

## レコード数
SELECT COUNT(*) FROM features;

## 条件検索
### 下記構造のjsonを想定
-- { 
-- 	"type": "Feature", 
-- 	"properties": { 
-- 		"MAPBLKLOT": "0001001", 
-- 		"BLKLOT": "0001001", 
-- 		"BLOCK_NUM": "0001", 
-- 		"LOT_NUM": "001", 
-- 		"FROM_ST": "0", 
-- 		"TO_ST": "0", 
-- 		"STREET": "UNKNOWN", 
-- 		"ST_TYPE": null, 
-- 		"ODD_EVEN": "E" 
-- 	}, 
-- 	"geometry": { 
-- 		"type": "Polygon", 
-- 		"coordinates": [ 
-- 			[ 
-- 				[ -122.422003528252475, 37.808480096967251, 0.0 ], 
-- 				[ -122.422076013325281, 37.808835019815085, 0.0 ], 
-- 				[ -122.421102174348633, 37.808803534992904, 0.0 ], 
-- 				[ -122.421062569067274, 37.808601056818148, 0.0 ], 
-- 				[ -122.422003528252475, 37.808480096967251, 0.0 ] 
-- 			] 
-- 		] 
-- 	} 
-- }
SELECT * FROM features WHERE JSON_CONTAINS(feature, '"O"', '$.properties.ODD_EVEN') ;
SELECT COUNT(*) FROM features WHERE JSON_CONTAINS(feature, '"O"', '$.properties.ODD_EVEN') ;

select * from features limit 3;

select COUNT(*)
from features
where JSON_CONTAINS(feature, '"D"', '$.properties.ODD_EVEN');

select 
		id,
		json_extract(feature, '$.type'),
		json_extract(feature, '$.properties.ODD_EVEN'),
		json_extract(feature, '$.geometry.coordinates')
	from features
	WHERE JSON_CONTAINS(feature, '"E"', '$.properties.ODD_EVEN')
	limit 3;


# ストアドルーチン
select SPECIFIC_NAME,ROUTINE_TYPE from information_schema.ROUTINES;

## ストアドプロシージャ(戻り値がない)
show procedure status;
show procedure status where Db='test';

CREATE PROCEDURE sample01()
	SELECT NOW();

call sample01();

CREATE PROCEDURE sample02( IN x INT )
	SELECT x + 1;

call sample02(3);



## ストアドファンクション(戻り値がある)
show function status;
show function status where Db='test';


-- select table_name, table_rows from information_schema.TABLES where table_schema = 'test';
-- select ROUTINE_NAME, ROUTINE_TYPE from information_schema.ROUTINES;


# ユーザー定義変数
SET @number = 3;
select @number;
SHOW USER_VARIABLES

SHOW VARIABLES where Variable_name LIKE 'number';

# 変数
SHOW VARIABLES where Variable_name LIKE 'character%';
SET group_concat_max_len = 65535;
SHOW VARIABLES where Variable_name LIKE 'group_concat_max_len';

