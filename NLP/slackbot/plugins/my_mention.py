#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from slackbot.bot import respond_to     # @botname: で反応するデコーダ
from slackbot.bot import listen_to      # チャネル内発言で反応するデコーダ
from slackbot.bot import default_reply  # 該当する応答がない場合に反応するデコーダ

from janome.tokenizer import Tokenizer  # 形態素解析ライブラリ
import os, re, json, random

# @respond_to('string')     bot宛のメッセージ
#                           stringは正規表現が可能 「r'string'」
# @listen_to('string')      チャンネル内のbot宛以外の投稿
#                           @botname: では反応しないことに注意
#                           他の人へのメンションでは反応する
#                           正規表現可能
# @default_reply()          DEFAULT_REPLY と同じ働き
#                           正規表現を指定すると、他のデコーダにヒットせず、
#                           正規表現にマッチするときに反応
#                           ・・・なのだが、正規表現を指定するとエラーになる？

# message.reply('string')   @発言者名: string でメッセージを送信
# message.send('string')    string を送信
# message.react('icon_emoji')  発言者のメッセージにリアクション(スタンプ)する
#                               文字列中に':'はいらない
@respond_to('メンション')
def mention_func(message):
    message.reply('私にメンションと言ってどうするのよ...') # メンション

@listen_to('リッスン')
def listen_func(message):
    message.send('誰かがリッスンと投稿したようね')      # ただの投稿
    message.reply('あなたね？')                           # メンション

@respond_to('かわいい')
def cool_func(message):
    message.reply('ありがとう。スタンプ押しとくね!')     # メンション
    message.react('+1')     # リアクション

# 自分で育てるbotモデル作成
dict_file = "chatbot-data.json"
dic = {}
tokenizer = Tokenizer() # janome

# 辞書があれば最初に読み込む
if os.path.exists(dict_file):
    dic = json.load(open(dict_file,"r"))

# 辞書に単語を記録する
def register_dic(words):
    global dic
    if len(words) == 0: return
    tmp = ["@"]
    for i in words:
        word = i.surface
        if word == "" or word == "\r\n" or word == "\n": continue
        tmp.append(word)
        if len(tmp) < 3: continue
        if len(tmp) > 3: tmp = tmp[1:]
        set_word3(dic, tmp)
        if word == "。" or word == "？":
            tmp = ["@"]
            continue
    # 辞書を更新するごとにファイルへ保存
    json.dump(dic, open(dict_file,"w", encoding="utf-8"))

# 三要素のリストを辞書として登録
def set_word3(dic, s3):
    w1, w2, w3 = s3
    if not w1 in dic: dic[w1] = {}
    if not w2 in dic[w1]: dic[w1][w2] = {}
    if not w3 in dic[w1][w2]: dic[w1][w2][w3] = 0
    dic[w1][w2][w3] += 1

# 文章を生成する
def make_sentence(head):
    if not head in dic: return ""
    ret = []
    if head != "@": ret.append(head)
    top = dic[head]
    w1 = word_choice(top)
    w2 = word_choice(top[w1])
    ret.append(w1)
    ret.append(w2)
    while True:
        if w1 in dic and w2 in dic[w1]:
            w3 = word_choice(dic[w1][w2])
        else:
            w3 = ""
        ret.append(w3)
        if w3 == "。" or w3 == "？" or w3 == "": break
        w1, w2 = w2, w3
    return "".join(ret)

def word_choice(sel):
    keys = sel.keys()
    return random.choice(list(keys))

# slackbotに返答させる
def make_reply(text):
    # まず単語を学習する
    if text[-1] != "。": text += "。"
    words = tokenizer.tokenize(text)
    register_dic(words)
    # 辞書に単語があれば、そこから話す
    for w in words:
        face = w.surface
        ps = w.part_of_speech.split(',')[0]
        if ps == "感動詞":
            return face + "。"
        if ps == "名詞" or ps == "形容詞":
            if face in dic: return make_sentence(face)
    return make_sentence("@")

# 入力メッセージを取得し、slackbotに返事をさせる
@default_reply()
def default_func(message):
    text = message.body['text']     # メッセージを取り出す
    res = make_reply(text)
    message.reply(res)