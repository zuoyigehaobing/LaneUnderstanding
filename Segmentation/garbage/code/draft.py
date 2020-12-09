import os
import cv2
import numpy as np
import pandas as pd
import keras
from keras.applications import imagenet_utils

from keras.layers import *
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import regularizers

# ================ Helper functions =======================
def imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ======================== End ============================


img_width, img_height, img_channels = 320, 240, 3
input_shape = (img_height, img_width, img_channels)
num_classes = 2
epoch = 1000
batch_size = 1
label_dim = img_width * img_height
# class weighting


# data loading, load a single image together with its ground truth
data, label = [], []

pic_path = r"../data/debug/train/0001TP_009210.png"
mask_path = pic_path.replace(".png", "_L.png").replace(r"/train/", r"/train_labels/")

img = cv2.imread(pic_path, cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

# normalize the input, ignored here [TBD]


# get the target mask, merge archway and road as lane mask for better visualization
lane_color = np.array([128, 64, 128])
archway_color = np.array([192, 0, 128])
mask[(np.any(mask != lane_color, axis=2)) & (np.any(mask != archway_color, axis=2))] = np.array([0, 0, 0])
mask[np.any(mask != 0, axis=2)] = 1

# reshape the image (or crop it) to fit the size specified as input_shape
img = cv2.resize(img, (img_width, img_height))
mask = cv2.resize(mask, (img_width, img_height))
mask = mask[:, :, 0]

# one-hot function
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


data.append(img)
label.append(one_hot(mask, num_classes))

train_X, train_y = np.array(data), np.array(label)
train_X = imagenet_utils.preprocess_input(train_X)

# check points
ModelCheckpt = keras.callbacks.ModelCheckpoint(
        filepath="../check_points/" + 'weights_epoch{epoch:d}.hdf5',
        save_best_only=False,
        period=200,
        verbose=1)

# create the model
def TrialNet(input_shape, classes):
    img_input = Input(shape=input_shape)
    # Encoder
    ##################################################################################################################################
    # Output shape = 128*128*3 -> 64*64*8
    conv_11 = Conv2D(8, (3, 3), activation='relu', padding='same',
                     name='conv_11', kernel_regularizer=regularizers.l2(0.01))(
        img_input)
    conv_12 = Conv2D(8, (3, 3), activation='relu', padding='same',
                     name='conv_12', kernel_regularizer=regularizers.l2(0.01))(
        conv_11)
    conv_13 = Conv2D(8, (3, 3), activation='relu', padding='same',
                     name='conv_13', kernel_regularizer=regularizers.l2(0.01))(
        conv_12)
    bn_1 = BatchNormalization()(conv_13)
    maxpl_1 = MaxPooling2D(pool_size=(2, 2))(bn_1)

    # Output shape = 64*64*8 -> 32*32*16
    maxpl_1 = Dropout(rate=0.2)(maxpl_1)
    conv_21 = Conv2D(16, (3, 3), activation='relu', padding='same',
                     name='conv_21', kernel_regularizer=regularizers.l2(0.01))(
        maxpl_1)
    conv_22 = Conv2D(16, (3, 3), activation='relu', padding='same',
                     name='conv_22', kernel_regularizer=regularizers.l2(0.01))(
        conv_21)
    conv_23 = Conv2D(16, (3, 3), activation='relu', padding='same',
                     name='conv_23', kernel_regularizer=regularizers.l2(0.01))(
        conv_22)
    bn_2 = BatchNormalization()(conv_23)
    maxpl_2 = MaxPooling2D(pool_size=(2, 2))(bn_2)

    # Output shape = 32*32*16 -> 16*16*32
    maxpl_2 = Dropout(rate=0.2)(maxpl_2)
    conv_31 = Conv2D(32, (3, 3), activation='relu', padding='same',
                     name='conv_31', kernel_regularizer=regularizers.l2(0.01))(
        maxpl_2)
    conv_32 = Conv2D(32, (3, 3), activation='relu', padding='same',
                     name='conv_32', kernel_regularizer=regularizers.l2(0.01))(
        conv_31)
    conv_33 = Conv2D(32, (3, 3), activation='relu', padding='same',
                     name='conv_33', kernel_regularizer=regularizers.l2(0.01))(
        conv_32)
    bn_3 = BatchNormalization()(conv_33)
    maxpl_3 = MaxPooling2D(pool_size=(2, 2), name='maxpl_3')(bn_3)

    # Output shape = 16*16*32 -> 8*8*64
    maxpl_3 = Dropout(rate=0.2)(maxpl_3)
    conv_41 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     name='conv_41', kernel_regularizer=regularizers.l2(0.01))(
        maxpl_3)
    conv_42 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     name='conv_42', kernel_regularizer=regularizers.l2(0.01))(
        conv_41)
    conv_43 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     name='conv_43', kernel_regularizer=regularizers.l2(0.01))(
        conv_42)
    bn_4 = BatchNormalization()(conv_43)
    ##################################################################################################################################

    # Decoder
    ##################################################################################################################################
    # Skip layer 1
    # Output shape = 16*16*32 -> 16*16*16
    '''
    skip_5 = Dropout(rate=0.2)(bn_3)
    conv_51 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_51', kernel_regularizer=regularizers.l2(0.01))(skip_5)
    conv_52 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_52', kernel_regularizer=regularizers.l2(0.01))(conv_51)
    conv_53 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_53', kernel_regularizer=regularizers.l2(0.01))(conv_52)
    bn_5    = BatchNormalization()(conv_53)
    # Skip layer 2
    # Output shape = 8*8*64 -> 16*16*16
    ups_6 = UpSampling2D(size=(2, 2))(bn_4)
    skip_6 = Dropout(rate=0.2)(ups_6)
    conv_61 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_61', kernel_regularizer=regularizers.l2(0.01))(skip_6)
    conv_62 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_62', kernel_regularizer=regularizers.l2(0.01))(conv_61)
    conv_63 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_63', kernel_regularizer=regularizers.l2(0.01))(conv_62)
    bn_6    = BatchNormalization()(conv_63)
    # Merge skip layers
    # Output shape = 16*16*16 + 16*16*16 -> 16*16*32
    merge_7 = Concatenate()([bn_5, bn_6])
    merge_7= Dropout(rate=0.2)(merge_7)
    conv_71 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_71', kernel_regularizer=regularizers.l2(0.01))(merge_7)
    conv_72 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_72', kernel_regularizer=regularizers.l2(0.01))(conv_71)
    conv_73 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_73', kernel_regularizer=regularizers.l2(0.01))(conv_72)
    bn_7    = BatchNormalization()(conv_73)
    '''

    # Output shape = 8*8*64 -> 16*16*32
    upsp_5 = Dropout(rate=0.2)(bn_4)
    conv_51 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     name='conv_51', kernel_regularizer=regularizers.l2(0.01))(
        upsp_5)
    conv_52 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     name='conv_52', kernel_regularizer=regularizers.l2(0.01))(
        conv_51)
    conv_53 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     name='conv_53', kernel_regularizer=regularizers.l2(0.01))(
        conv_52)
    bn_5 = BatchNormalization()(conv_53)

    # Output shape = 16*16*32 -> 32*32*16
    upsp_6 = UpSampling2D(size=(2, 2))(bn_5)
    upsp_6 = Dropout(rate=0.2)(upsp_6)
    conv_61 = Conv2D(32, (3, 3), activation='relu', padding='same',
                     name='conv_61', kernel_regularizer=regularizers.l2(0.01))(
        upsp_6)
    conv_62 = Conv2D(32, (3, 3), activation='relu', padding='same',
                     name='conv_62', kernel_regularizer=regularizers.l2(0.01))(
        conv_61)
    conv_63 = Conv2D(32, (3, 3), activation='relu', padding='same',
                     name='conv_63', kernel_regularizer=regularizers.l2(0.01))(
        conv_62)
    bn_6 = BatchNormalization()(conv_63)

    # Output shape = 32*32*16 -> 64*64*16
    upsp_8 = UpSampling2D(size=(2, 2))(bn_6)
    upsp_8 = Dropout(rate=0.2)(upsp_8)
    conv_81 = Conv2D(16, (3, 3), activation='relu', padding='same',
                     name='conv_81', kernel_regularizer=regularizers.l2(0.01))(
        upsp_8)
    conv_82 = Conv2D(16, (3, 3), activation='relu', padding='same',
                     name='conv_82', kernel_regularizer=regularizers.l2(0.01))(
        conv_81)
    conv_83 = Conv2D(16, (3, 3), activation='relu', padding='same',
                     name='conv_83', kernel_regularizer=regularizers.l2(0.01))(
        conv_82)
    bn_8 = BatchNormalization()(conv_83)

    # Output shape = 32*32*16 -> 128*128*8
    upsp_9 = UpSampling2D(size=(2, 2))(bn_8)
    upsp_9 = Dropout(rate=0.2)(upsp_9)
    conv_91 = Conv2D(8, (3, 3), activation='relu', padding='same',
                     name='conv_91', kernel_regularizer=regularizers.l2(0.01))(
        upsp_9)
    conv_92 = Conv2D(8, (3, 3), activation='relu', padding='same',
                     name='conv_92', kernel_regularizer=regularizers.l2(0.01))(
        conv_91)
    conv_93 = Conv2D(8, (3, 3), activation='relu', padding='same',
                     name='conv_93', kernel_regularizer=regularizers.l2(0.01))(
        conv_92)
    bn_9 = BatchNormalization()(conv_93)

    cl_10 = Conv2D(classes, (1, 1), padding="valid")(bn_9)
    rs_10 = Reshape((input_shape[0] * input_shape[1], classes))(cl_10)
    output_10 = Activation("softmax")(rs_10)
    model = Model(img_input, output_10)
    return model


model = TrialNet(input_shape, num_classes)
#model = keras.models.load_model('./history/weights_epoch195.hdf5')
optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


model.fit(train_X, train_y, batch_size=batch_size, callbacks=[ModelCheckpt], epochs=epoch, verbose=2,
              validation_data=(train_X, train_y), shuffle=False)


