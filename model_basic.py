#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils

def SegNet(input_shape=(360, 480, 3), classes=12):
    ### @ https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    
    
    ## Encoder
    # 1
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    print(x)

    # 2
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    print(x)

    # 3
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    print(x)

    # 4
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
#    x = MaxPooling2D(pool_size=(2, 2))(x)
    print(x)

    ## Decoder
    # -4
#    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    #x = Activation("relu")(x)
    print(x)

    # -3
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    #x = Activation("relu")(x)
    print(x)

    # -2
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    #x = Activation("relu")(x)
    print(x)

    # -1
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    #x = Activation("relu")(x)
    print(x)

    # 出力
    x = Conv2D(classes, (1, 1), padding="valid")(x)
    x = Reshape((input_shape[0] * input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    print(x)
    
    model = Model(img_input, x)
    
    return model