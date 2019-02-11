#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

## ディレクトリ変更
os.chdir("/Users/tomoyuki/python_workspace/openCV")
os.getcwd()

# 画像読込み
img_name = 'road2.jpg'
img = cv2.imread("input/"+img_name,cv2.IMREAD_COLOR)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# 道路マーカーを白，それ以外を黒に変換
tol=70
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (img[i,j,0]>(2-tol) and img[i,j,0]<(2+tol) and
            img[i,j,1]>(70-tol) and img[i,j,1]<(70+tol) and
            img[i,j,2]>(253-tol) and img[i,j,2]<(253+tol)):
            img[i,j,:]=np.array([255,255,255])

        else :
            img[i,j,:]=np.array([0,0,0])

plt.imshow(img)            
img.shape
cv2.imwrite("input/roadmarker2.jpg",img)
