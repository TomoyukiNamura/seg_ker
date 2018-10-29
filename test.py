#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import dataset


input_shape = (360, 480, 3)
classes = 12

data_path = './CamVid/'
test_file = 'test.txt'

trained_model_path = './trained_models/'
trained_model_name = 'seg_basic_epoch1.h5'

output_file = './output/'



# 結果画像の保存用関数
def writeImage(image, filename):
    """ label data to colored image """
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0,12):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)


## set gpu usage
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.8))
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False))
config = tf.ConfigProto(device_count={'GPU': 0})


with tf.Session(config=config) as session:
    
    # データセット読み込み用インスタンスの初期化
    ds = dataset.Dataset(data_shape=input_shape, classes=classes, data_path=data_path, test_file=test_file)
    
    # 評価データ(入力画像とアノテーション)の読み込み
    print("loading test data...")
    test_X, test_y = ds.load_data('test') # need to implement, y shape is (None, 360, 480, classes)
    test_X = ds.preprocess_inputs(test_X)
    test_Y = ds.reshape_labels(test_y)
    print("end")
    
    # モデル読み込み
    print("loading SegNet...")
    model = keras.models.load_model(trained_model_path + trained_model_name)
    print("end")
    
    # 評価を実行
    print("running SegNet...")
    probs = model.predict(test_X, batch_size=1)
    print("end")
    
    # 評価結果を保存
    for i in range(probs.shape[0]):
        prob = probs[i].reshape((input_shape[0], input_shape[1], classes)).argmax(axis=2)
        writeImage(prob, output_file + f'result_{i}.png')
