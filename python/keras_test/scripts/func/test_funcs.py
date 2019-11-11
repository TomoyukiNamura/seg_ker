#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

 testスクリプト用関数 

"""


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# MAE計算
def calcMAE(df_truth, df_pred):
    tmp_bool = df_truth.isna() == False
    if np.any(tmp_bool)==True:
        mae = mean_absolute_error(y_true=df_truth[tmp_bool], y_pred=df_pred[tmp_bool])
    else:
        mae = np.nan
    return mae

# 訓練データ，予測結果プロット
def PlotTruthPred(df_train, df_truth, df_pred, inspects_dict=None, ylim=None, r_plot_size=1, output_dir=None, file_name=""):
        
    # MAE計算
    mae = calcMAE(df_truth, df_pred)
    
    # プロット
    plt.rcParams["font.size"] = 10*r_plot_size
    plt.rcParams['figure.figsize'] = [6.0*r_plot_size, 4.0*r_plot_size]
    
    plt.plot(df_train, label="train", color="black")
    plt.plot(df_truth, label="truth", color="blue")
    plt.plot(df_pred, label="pred", color="red")
    
    if ylim!=None:
        plt.ylim(ylim)
        
    if inspects_dict!=None:
        xlabel = ""
        for key in list(inspects_dict.keys()):
            xlabel = xlabel + f"{key}:{inspects_dict[key]}  "
        plt.xlabel(xlabel)
        
    plt.grid()
    plt.title(file_name + f"    mae: {round(mae, 4)}")    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    if output_dir!=None and file_name!="":
        if os.path.exists(output_dir)==False:
            os.mkdir(output_dir)
            
        plt.savefig(f"{output_dir}/{file_name}.jpg", bbox_inches='tight')
    
    plt.show()
    
# MAEプロット
def plotTotalMAE(mae_dict, ylim=None, r_plot_size=1, output_dir=None):
        
    mae_vector = np.array(list(mae_dict.values()))
    mae_vector = mae_vector[~np.isnan(mae_vector)]
    
    plt.rcParams["font.size"] = 10*r_plot_size
    plt.rcParams['figure.figsize'] = [6.0*r_plot_size, 4.0*r_plot_size]
    
    plt.plot(mae_vector, color="blue")
    plt.grid()
    plt.title(f"total MAE : {np.round(np.mean(mae_vector),4)}")
    if ylim!=None:
        plt.ylim(ylim)
    
    if output_dir!=None:
        if os.path.exists(output_dir)==False:
            os.mkdir(output_dir)
        plt.savefig(f"{output_dir}/total_MAE.jpg", bbox_inches='tight')
        
    plt.show()
    

