#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.chdir('/Users/tomoyuki/python_workspace/SegNet_tensorflow')
import tensorflow as tf
import keras
from model import SegNet
import dataset


input_shape = (360, 480, 3)
classes = 12
epochs = 1
batch_size = 1

data_path = './CamVid/'
train_file = 'train.txt'
val_file = 'val.txt'

log_file_path = './logs/'
trained_model_path = './trained_models/'
trained_model_name = 'segnet_basic.h5'

class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]



## set gpu usage
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.8))
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False))
config = tf.ConfigProto(device_count={'GPU': 0})

with tf.Session(config=config) as session:
    # keras
    keras.backend.tensorflow_backend.set_session(session)
    
    # データセット読み込み用インスタンスの初期化
    ds = dataset.Dataset(data_shape=input_shape, classes=classes, data_path=data_path, train_file=train_file, val_file=val_file)
    
    # 訓練データ(入力画像とアノテーション)の読み込み
    print("loading train data...")
    train_X, train_y = ds.load_data('train') # need to implement, y shape is (None, 360, 480, classes)
    train_X = ds.preprocess_inputs(train_X)    
    train_Y = ds.reshape_labels(train_y)
    print("end")
    
    # バリデーションデータ(入力画像とアノテーション)の読み込み
    print("loading validation data...")
    val_X, val_y = ds.load_data('val') # need to implement, y shape is (None, 360, 480, classes)
    val_X = ds.preprocess_inputs(val_X)
    val_Y = ds.reshape_labels(val_y)
    print("end")
    
    # tensorboadに学習過程?を表示する処理
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_file_path, histogram_freq=1, write_graph=True, write_images=True)
    
    # モデル作成
    print("creating model...")
    model = SegNet(input_shape=input_shape, classes=classes)
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
    model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs,
              verbose=1, class_weight=class_weighting , validation_data=(val_X, val_Y), shuffle=True
              , callbacks=[tb_cb])
    model.save(trained_model_path + trained_model_name)
    print("end")


