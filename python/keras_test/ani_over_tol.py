import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2


train = pd.read_csv(f"train_over_tol.csv",index_col=0)
#test = pd.read_csv(f"test.csv",index_col=0)
pred_ARIMA = pd.read_csv(f"pred_ARIMA_over_tol.csv",index_col=0)
#MAE_ARIMA = pd.read_csv(f"MAE_ARIMA.csv")
#pred_mean = pd.read_csv(f"pred_mean.csv",index_col=0)

milage_list = list(train.columns)

fig = plt.figure()

def plot(data):
    plt.cla()
    milage = milage_list[data]

    #mae = np.round(float(MAE_ARIMA["MAE"][MAE_ARIMA["milage"]==milage]),3)
    mae = 0
    plt.plot(train[milage], color="black")
    #plt.plot(test[milage], color="blue")
    plt.plot(pred_ARIMA[milage], color="red")
    plt.grid();plt.title(f"{data}_{milage}  MAE:{mae}");plt.ylim([-15, 15])

# アニメーションを作成する。
anim = animation.FuncAnimation(fig, plot, frames=len(milage_list))

# Figure を表示する。
plt.show()

