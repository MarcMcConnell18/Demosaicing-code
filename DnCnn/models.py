import numpy as np
from keras.models import Model
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Lambda,Subtract,Add, Dropout


def DnCNN():
    
    inpt = Input(shape=(None,None,3))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(x)
    model = Model(inputs=inpt, outputs=x)
    
    return model

