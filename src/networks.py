# import os
# import skimage.io as io
# import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D
from tensorflow.keras.layers import Activation, MaxPool2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# from keras.optimizers import schedules
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras


def unet(pretrained_weights = None, input_size = (60,60,1), output_classes = 1):
    """
    This UNet architecture is based on this paper:
        "Computer vision-based concrete crack detection using U-net fully convolutional networks"
        Liu et al., 2019, J. Automation in Construction    
    """
    inputs = Input(input_size)
    # convolutional block 1
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='conv1')(inputs)
    bn1 = BatchNormalization(3, name='bn1')(conv1) 
    L1 = Activation('relu')(bn1)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='conv2')(L1)
    bn2 = BatchNormalization(3, name='bn2')(conv2) 
    L2 = Activation('relu')(bn2)
    pool1 = MaxPool2D(pool_size=(2,2), name='pool1')(L2)

    # convolutional block 2
    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='conv3')(pool1)
    bn3 = BatchNormalization(3, name='bn3')(conv3) 
    L3 = Activation('relu')(bn3)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='conv4')(L3)
    bn4 = BatchNormalization(3, name='bn4')(conv4) 
    L4 = Activation('relu')(bn4)
    pool2 = MaxPool2D(pool_size=(2,2), name='pool2')(L4)

    # convolutional block 3
    conv5 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='conv5')(pool2)
    bn5 = BatchNormalization(3, name='bn5')(conv5) 
    L5 = Activation('relu')(bn5)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='conv6')(L5)
    bn6 = BatchNormalization(3, name='bn6')(conv6) 
    L6 = Activation('relu')(bn6)
    pool3 = MaxPool2D(pool_size=(2,2), name='pool3')(L6)

    # convolutional block 4
    conv7 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='conv7')(pool3)
    bn7 = BatchNormalization(3, name='bn7')(conv7) 
    L7 = Activation('relu')(bn7)
    conv8 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='conv8')(L7)
    bn8 = BatchNormalization(3, name='bn8')(conv8) 
    L8 = Activation('relu')(bn8)
    pool4 = MaxPool2D(pool_size=(2,2), name='pool4')(L8)

    # convolution block 5
    conv9 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', name='conv9')(pool4)
    bn9 = BatchNormalization(3, name='bn9')(conv9) 
    L9 = Activation('relu')(bn9)
    conv10 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', name='conv10')(L9)
    bn10 = BatchNormalization(3, name='bn10')(conv10) 
    L10 = Activation('relu')(bn10)
    up5 = Conv2DTranspose(512, (2,2), strides=2, kernel_initializer='he_normal', name='up5')(L10)

    # convolutional block 6
    merge6 = concatenate([L8,up5], axis=3)
    conv11 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='conv11')(merge6)
    bn11 = BatchNormalization(3, name='bn11')(conv11) 
    L11 = Activation('relu')(bn11)
    conv12 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='conv12')(L11)
    bn12 = BatchNormalization(3, name='bn12')(conv12) 
    L12 = Activation('relu')(bn12)
    up6 = Conv2DTranspose(256, (2,2), strides=2, kernel_initializer='he_normal', name='up6')(L12)

    # convolutional block 7
    merge7 = concatenate([L6,up6], axis=3)
    conv13 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='conv13')(merge7)
    bn13 = BatchNormalization(3, name='bn13')(conv13) 
    L13 = Activation('relu')(bn13)
    conv14 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='conv14')(L13)
    bn14 = BatchNormalization(3, name='bn14')(conv14) 
    L14 = Activation('relu')(bn14)
    up7 = Conv2DTranspose(128, (2,2), strides=2, kernel_initializer='he_normal', name='up7')(L14)

    # convolutional block 8
    merge8 = concatenate([L4,up7], axis=3)
    conv15 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='conv15')(merge8)
    bn15 = BatchNormalization(3, name='bn15')(conv15) 
    L15 = Activation('relu')(bn15)
    conv16 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='conv16')(L15)
    bn16 = BatchNormalization(3, name='bn16')(conv16) 
    L16 = Activation('relu')(bn16)
    up8 = Conv2DTranspose(64, (2,2), strides=2, kernel_initializer='he_normal', name='up8')(L16)

    # convolutional block 9
    merge9 = concatenate([L2,up8], axis=3)
    conv17 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='conv17')(merge9)
    bn17 = BatchNormalization(3, name='bn17')(conv17) 
    L17 = Activation('relu')(bn17)
    conv18 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='conv18')(L17)
    bn18 = BatchNormalization(3, name='bn18')(conv18) 
    L18 = Activation('relu')(bn18)
    conv19 = Conv2D(output_classes, 1, activation="softmax", kernel_initializer='he_normal', name='output')(L18)


    model = Model(inputs = inputs, outputs = conv19)

    #lr_schedule = schedules.ExponentialDecay(initial_learning_rate = 1e-3, decay_steps = 600, decay_rate = .1, staircase = True)

    model.compile(optimizer = Adam(learning_rate = 1e-3),
                  loss = 'categorical_crossentropy',
                  metrics = ["accuracy"])

    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    print("Comment: softmax")

    return model


