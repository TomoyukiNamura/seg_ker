#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import zscore

## 初期設定
color_list = ['red','blue','green','gray','yellow','magenta','cyan']
color_list_RGB = [(255,0,0),(0,0,255),(0,255,0),(125,125,125),(255,255,0),(255,0,255),(0,255,255)]

    
## 関数 
def isInVanishingPointField(vanish_point,tol_vanish,alpha,beta):
    # y=alpha_1+beta*x
    tmp = alpha+beta*(vanish_point[0]-tol_vanish)
    bool_1 = (vanish_point[1]-tol_vanish)<=tmp and (vanish_point[1]+tol_vanish)>=tmp
    
    tmp = alpha+beta*(vanish_point[0]+tol_vanish)
    bool_2 = (vanish_point[1]-tol_vanish)<=tmp and (vanish_point[1]+tol_vanish)>=tmp
    
    # x=(y-alpha)/beta
    tmp = ((vanish_point[1]-tol_vanish)-alpha)/beta
    bool_3 = (vanish_point[0]-tol_vanish)<=tmp and (vanish_point[0]+tol_vanish)>=tmp
    
    tmp = ((vanish_point[1]+tol_vanish)-alpha)/beta
    bool_4 = (vanish_point[0]-tol_vanish)<=tmp and (vanish_point[0]+tol_vanish)>=tmp
    
    return any([bool_1,bool_2,bool_3,bool_4])


## ディレクトリ変更
os.chdir("/Users/tomoyuki/python_workspace/openCV")
os.getcwd()

# 画像読込み
img_name = 'roadmarker2.jpg'
org = cv2.imread("input/"+img_name,cv2.IMREAD_COLOR)
gray = cv2.cvtColor(org,cv2.COLOR_BGR2GRAY)
plt.imshow(gray, 'gray')


## 1_エッジ検出
edges = cv2.Canny(gray,50,150,apertureSize = 5)
plt.imshow(edges, 'gray')


## 2_ハフ変換による直線抽出
lines = cv2.HoughLines(edges,1,np.pi/180,30)
#lines = cv2.HoughLines(gray,1,np.pi/180,100)
lines.shape

img = copy.copy(org)
for i in range(lines.shape[0]):
    for rho,theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

plt.imshow(img)



## 3_消失点領域を用いた直線の剪定
# 消失点座標と領域閾値を指定
vanish_point=[322.4,220.6]
tol_vanish=50

lines_new = []

img = copy.copy(org)
for i in range(lines.shape[0]):
    for rho,theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        # 切片・傾きの計算
        beta = (y2-y1)/(x2-x1)
        alpha = y2-beta*x2
        
        # ある線が消失点領域に入っているか計算
        is_in_vanish = isInVanishingPointField(vanish_point, tol_vanish , alpha=alpha, beta=beta)

        # 消失点領域に入っていれば線を取得
        if is_in_vanish:
            lines_new.append(lines[i])
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
            
lines_new = np.array(lines_new)       
plt.imshow(img)




## 4_ハフ変換結果をクラスタリング
# ハフ変換結果(ラジアンの度数法への変換，標準化を実施)をクラスタリング
rho_old   = lines_new[:,0,0]
theta_old = lines_new[:,0,1]

array_hough = np.array([zscore(rho_old),
                        zscore(np.rad2deg(theta_old))
                        ]).T

k = 4
pred = KMeans(n_clusters=k).fit_predict(array_hough)

## クラスタリング結果プロット
#color=[]
#for i in range(pred.shape[0]):
#    color.append(color_list[pred[i]])
#df_hough = pd.DataFrame({ 'rho' : rho_old, 'theta' : theta_old })
#df_hough.plot(kind='scatter', x=u'rho', y=u'theta',c=color)


# 結果プロット
img = copy.copy(org)
for i in range(lines_new.shape[0]):
    for rho,theta in lines_new[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),color_list_RGB[pred[i]],2)

plt.imshow(img)



## 5_クラスタ毎の代表値を計算し，白線とみなす(代表値は中央値を採用)
rho_new = []
theta_new = []

for i in range(k):
    rho_new.append(np.median(rho_old[pred == i]))
    theta_new.append(np.median(theta_old[pred == i]))
    
rho_new = np.array(rho_new)
theta_new = np.array(theta_new)


# 結果プロット
img = copy.copy(org)
for i in range(rho_new.shape[0]):
    rho = rho_new[i]
    theta = theta_new[i]

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    
    cv2.line(img,(x1,y1),(x2,y2),color_list_RGB[i],2)
    
    
#    # 切片・傾きの表示
#    beta = (y2-y1)/(x2-x1)
#    alpha = y2-beta*x2
#    print(color_list[i])
#    print(color_list_RGB[i])
#    print("beta:",round(beta,2))
#    print("alpha:",round(alpha,2))
#    print("")
    
    
plt.imshow(img)








        


## 2本の白線の切片，傾きを取得
#beta_1 = -1.11
#alpha_1 = 578.54
#
#beta_2 = 1.8
#alpha_2 = -359.76
#
#
## 消失点の計算
#x_vanish = (alpha_1-alpha_2)/(beta_2-beta_1)
#y_vanish = (beta_2*alpha_1-beta_1*alpha_2)/(beta_2-beta_1)
#vanish_point = [x_vanish,y_vanish]