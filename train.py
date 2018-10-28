#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.chdir('/Users/tomoyuki/python_workspace/SegNet_tensorflow')

import glob
import numpy as np
import keras

import cv2
import matplotlib.pyplot as plt

#from model import SegNet
from model_basic import SegNet

import dataset

input_shape = (360, 480, 3)
classes = 12
epochs = 1
batch_size = 1
log_filepath='./logs/'

data_shape = 360*480

class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

## set gpu usage
import tensorflow as tf
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.8))
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False))
config = tf.ConfigProto(device_count={'GPU': 0})

with tf.Session(config=config) as session:
    #session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

    
    # 入力画像とアノテーションの読み込み
    print("loading data...")
    ds = dataset.Dataset(classes=classes)
    train_X, train_y = ds.load_data('train') # need to implement, y shape is (None, 360, 480, classes)
    
    ## 画像を表示できない
    #tmp_img = np.array(train_X[1])
    #plt.imshow(tmp_img)
    
    # 入力画像をtensorflowの入力形式に変更?(-1~1ではないっぽい)
    train_X = ds.preprocess_inputs(train_X)
    
    # アノテーション画像をリシェイプ
    train_Y = ds.reshape_labels(train_y)
    
    print("input data shape...", train_X.shape)
    print("input label shape...", train_Y.shape)
    
    test_X, test_y = ds.load_data('test') # need to implement, y shape is (None, 360, 480, classes)
    test_X = ds.preprocess_inputs(test_X)
    test_Y = ds.reshape_labels(test_y)
    
    # tensorboadに学習過程?を表示する処理
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1, write_graph=True, write_images=True)
    
    # モデル作成
    print("creating model...")
    model = SegNet(input_shape=input_shape, classes=classes)
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
    
    model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs,
              verbose=1, class_weight=class_weighting , validation_data=(test_X, test_Y), shuffle=True
              , callbacks=[tb_cb])
    
    model.save('trained_models/seg.h5')

print("彡(＾)(＾)")