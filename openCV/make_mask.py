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
img_name = 'no4_mask.jpg'
img = cv2.imread("input/"+img_name,cv2.IMREAD_COLOR)

# BGR->RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# 
img[img != 255]=0
plt.imshow(img)

cv2.imwrite('input/mask_'+img_name,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
