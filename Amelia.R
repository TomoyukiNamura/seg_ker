# 実行環境：R version 3.3.2 (2016-10-31)　→　正常な動作確認
# 実行環境：R version 3.5.1 (2018-07-02)　→　正常な動作確認

#install.packages("Amelia")
library("Amelia")


# テストデータ
data <- data.frame(
          x = c(0,1,0,0,1,0,1,1,0,0,1,0,1,1,0,1),
          y = c(1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,1)
        )

# テストデータの一部を欠損させる
data[1,"x"] = NA
data[3,"y"] = NA


# EMB実行
set.seed(10) # シード固定(なくてもいい)
data_imputed <- amelia(data,   # 欠損ありデータ（data.frame形式）
                       m = 10) # ブートストラップ標本抽出による欠損補完を行う回数
                       
# EMBによる欠損補完結果
data_imputed$imputations # m回の欠損補完結果を表示
data_imputed$imputations$imp4 # 特定の欠損補完結果(4回目)を表示

