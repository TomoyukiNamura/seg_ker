#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### 初期処理 =======================================================

## ディレクトリ変更
import os
os.chdir('/Users/tomoyuki/python_workspace/NLP')

# パッケージ読み込み
from janome.tokenizer import Tokenizer
import json

# テキストファイルを読み込む
sjis = open('input/review_mei.txt', 'rb').read()
text = sjis.decode('utf_8')

# テキストを形態素解析読み込みます
t = Tokenizer()
words = t.tokenize(text)


# 辞書を生成します
def make_dic(words):
    tmp = ["@"]
    dic = {}
    for i in words:
        word = i.surface
        if word == "" or word == "\r\n" or word == "\n": continue
        tmp.append(word)
        if len(tmp) < 3: continue
        if len(tmp) > 3: tmp = tmp[1:]
        set_word3(dic, tmp)
        if word == "。":
            tmp = ["@"]
            continue
    return dic

# 三要素のリストを辞書として登録しています
def set_word3(dic, s3):
    w1, w2, w3 = s3
    if not w1 in dic: dic[w1] = {}
    if not w2 in dic[w1]: dic[w1][w2] = {}
    if not w3 in dic[w1][w2]: dic[w1][w2][w3] = 0
    dic[w1][w2][w3] += 1

dic = make_dic(words)
json.dump(dic, open("output/markov-review_mei.json", "w", encoding="utf-8"))


# 記事を作文します
def make_sentence(dic):
    ret = []
    if not "@" in dic: return "no dic"
    top = dic["@"]
    w1 = word_choice(top)
    w2 = word_choice(top[w1])
    ret.append(w1)
    ret.append(w2)
    while True:
        w3 = word_choice(dic[w1][w2])
        ret.append(w3)
        if w3 == "。": break
        w1, w2 = w2, w3
    return "".join(ret)
    
for i in range(20):
    s = make_sentence(dic)
    print(s)