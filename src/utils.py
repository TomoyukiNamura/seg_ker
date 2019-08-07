#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import r2_score

# 出力フォルダ作成関数
def make_new_folder(folder_name):
    if os.path.exists(folder_name)==False:
        os.mkdir(folder_name)
        return True
    else:
        return False
    
# 予測結果表示
def plot_pred(true, pred, margin_rate=0.1, s_rate=0.3, alpha=0.5):
    margin = (np.max(true) - np.min(true)) * margin_rate
    plt.plot(true, true, c="r", alpha=alpha)
    plt.scatter(true, pred, s=plt.rcParams['lines.markersize']**2 * s_rate, alpha=alpha)
    plt.title("r2 : "+str(r2_score(true, pred)))
    plt.grid()
    plt.xlim([np.min(true)-margin, np.max(true)+margin])
    plt.ylim([np.min(true)-margin, np.max(true)+margin])
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.show()

    
## アニメーション
#def animation_plot(df_target, df_data, margin_rate=0.1, s_rate=0.3, alpha=0.5):
#    margin = (np.max(df_target) - np.min(df_target)) * margin_rate
#    columns = list(df_data.columns)
#    fig = plt.figure()
#    
#    def _plot(data):
#        plt.cla()
#        column = columns[data]
#        plt.scatter(df_data.loc[:,column], df_target, s=plt.rcParams['lines.markersize']**2 * s_rate, alpha=alpha)
#        plt.grid()
#        plt.title(f"{column}")
#        plt.ylim([np.min(df_target)-margin, np.max(df_target)+margin])
#        plt.xlabel(f"{column}")
#
#    anim = animation.FuncAnimation(fig, _plot, frames=len(columns))
#    plt.show()
        