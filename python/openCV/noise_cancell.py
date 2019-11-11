#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

## ディレクトリ変更
os.chdir("/Users/tomoyuki/openCV")
os.getcwd()

# 画像読込み
img_name = 'no2.jpg'
img = cv2.imread("input/"+img_name,cv2.IMREAD_COLOR)

# BGR->RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)



## 画像平滑化(ぼかし？)
blur = cv2.blur(img,(10,10))
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(blur)
plt.show()
cv2.imwrite('output/blur_'+img_name,cv2.cvtColor(blur, cv2.COLOR_RGB2BGR))


## ノイズ除去
# Non-Local Means Denoising
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()
cv2.imwrite("output/NLMD_"+img_name,cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))


## 画像鮮鋭化
# アンシャープマスキング
i = 2
k = 1
kernel = np.ones((i,i),np.float32)/(i*i)
dst = cv2.filter2D(img,-1,kernel)
res = img + (img - dst)*k
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(res)
plt.show()
cv2.imwrite('output/UM_'+img_name,cv2.cvtColor(res, cv2.COLOR_RGB2BGR))



## 画像のInpainting
img_name = 'no4.jpg'
img = cv2.imread("input/"+img_name,cv2.IMREAD_COLOR)
mask = cv2.imread("input/mask_"+img_name,cv2.IMREAD_COLOR)
mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
plt.imshow(mask)

dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
plt.imshow(dst)
cv2.imwrite('output/Inpainting_'+img_name,dst)
