#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
import re
import traceback


# 括弧を削除する関数
def removeBracket(string, rule_list=[r'\(.+?\)',r'\{.+?\}']):
    try:
        result = deepcopy(string)
        remove_target = []
        for rule in rule_list:
            remove_target.extend(re.findall(rule, result))
        
    except:
        return np.nan, ""
    
    for remove_str in remove_target:
        result = result.replace(remove_str, "")
        
    return result, "/".join(remove_target)




# 文字列のタイプ(カタカナ，ひらがな)を判定する関数
def checkCharacterType(string ,ctype="hira", black_list = ["ー"]):
    """
    ひらがな：ctype=r'[\u3041-\u3094]'
    カタカナ：ctype=r'[\u30A1-\u30F4]+'
    """
    
    try:
        if ctype=="hira":
            regular_exp = r'[\u3041-\u3094]'
            
        elif ctype=="kata":
            regular_exp = r'[\u30A1-\u30F4]+'
            
        else:
            regular_exp = None
        
        re_katakana = re.compile(regular_exp)
        
    except:
        traceback.print_exc()
        return None
    
    # 調査対象外文字の除外
    for black_str in black_list:
        string = "".join(string.split(black_str))
    
    return re_katakana.fullmatch(string)!=None




# 特定の文字タイプ(カタカナ，ひらがな)以外の文字を削除する関数
def removeCharInString(string ,ctype="hira", black_list = ["ー"]):  
    """
    ひらがな：ctype=r'[\u3041-\u3094]'
    カタカナ：ctype=r'[\u30A1-\u30F4]+'
    """
    
    try:
        if ctype=="hira":
            regular_exp = r'[\u3041-\u3094]'
            
        elif ctype=="kata":
            regular_exp = r'[\u30A1-\u30F4]+'
            
        else:
            regular_exp = None
        
        re_katakana = re.compile(regular_exp)
        
    except:
        traceback.print_exc()
        return None
    
    # 新しい文字列作成
    new_string = ""
    for i in range(len(string)):
        if (re_katakana.fullmatch(string[i])!=None) or (string[i] in black_list):
            new_string = new_string + string[i]
            
    return new_string




