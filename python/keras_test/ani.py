import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2


raw0 = pd.read_csv(f"raw0.csv")
#tmp_raw0_median = pd.read_csv(f"tmp_raw0_median.csv")
#diff_prior_treated = pd.read_csv(f"diff_prior_treated.csv")
raw0_prior_treated = pd.read_csv(f"raw0_prior_treated.csv")
milage_list = list(raw0.columns)

fig = plt.figure()

def plot(data):
    plt.cla()
    milage = milage_list[data]
    plt.plot(raw0[milage], color="blue",alpha=0.5)
   # plt.plot(tmp_raw0_median[milage], color="red",alpha=0.5)
   # plt.plot(diff_prior_treated[milage], color="orange",alpha=0.5)
    plt.plot(raw0_prior_treated[milage], color="green",alpha=1)
    plt.grid();plt.title(f"{data}_{milage}");plt.ylim([-15, 15])

# アニメーションを作成する。
anim = animation.FuncAnimation(fig, plot, frames=len(milage_list))

# Figure を表示する。
plt.show()

