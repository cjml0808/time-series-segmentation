from tensorflow.keras import layers
import tensorflow.keras as k
from tensorflow.keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from torch import nn
from tensorflow.keras import regularizers
import numpy as np
import torch

class Unet_model(object):
    def __init__(self):
        self.windows = 64
        self.final_act = 'sigmoid'
        self.filter = [64, 128, 256, 512, 1024]

    # .................................................................................................................
    def Unet(self):
        inputs = layers.Input((self.windows, 33))
        x1 = layers.Conv1D(64, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(inputs)
        # x1 = layers.BatchNormalization()(x1)
        # x1 = layers.ReLU()(x1)
        x1 = layers.Conv1D(64, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x1)
        # x1 = layers.BatchNormalization()(x1)
        # x1 = layers.ReLU()(x1)

        x2 = layers.MaxPooling1D()(x1)
        x2 = layers.Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x2)
        # x2 = layers.BatchNormalization()(x2)
        # x2 = layers.ReLU()(x2)
        x2 = layers.Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x2)
        # x2 = layers.BatchNormalization()(x2)
        # x2 = layers.ReLU()(x2)

        x3 = layers.MaxPooling1D()(x2)
        x3 = layers.Conv1D(256, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x3)
        # x3 = layers.BatchNormalization()(x3)
        # x3 = layers.ReLU()(x3)
        x3 = layers.Conv1D(256, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x3)
        # x3 = layers.BatchNormalization()(x3)
        # x3 = layers.ReLU()(x3)

        x4 = layers.MaxPooling1D()(x3)
        x4 = layers.Conv1D(512, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x4)
        # x4 = layers.BatchNormalization()(x4)
        # x4 = layers.ReLU()(x4)
        x4 = layers.Conv1D(512, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x4)
        # x4 = layers.BatchNormalization()(x4)
        # x4 = layers.ReLU()(x4)
        x4 = layers.Dropout(0.5)(x4)

        x5 = layers.MaxPooling1D()(x4)
        x5 = layers.Conv1D(1024, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x5)
        # x2 = layers.BatchNormalization()(x2)
        # x2 = layers.ReLU()(x2)
        x5 = layers.Conv1D(1024, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x5)
        # x2 = layers.BatchNormalization()(x2)
        # x2 = layers.ReLU()(x2)
        x5 = layers.Dropout(0.5)(x5)

        x4_up = layers.UpSampling1D()(x5)
        x4_up = layers.Conv1D(512, kernel_size=2, padding='same', kernel_initializer='he_normal', activation='relu')(x4_up)
        # x4_up = layers.BatchNormalization()(x4_up)
        # x4_up = layers.ReLU()(x4_up)
        x4_up = x4 + x4_up
        x4_up = layers.Conv1D(512, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x4_up)
        # x4_up = layers.BatchNormalization()(x4_up)
        # x4_up = layers.ReLU()(x4_up)
        x4_up = layers.Conv1D(512, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x4_up)
        # x4_up = layers.BatchNormalization()(x4_up)
        # x4_up = layers.ReLU()(x4_up)

        x3_up = layers.UpSampling1D()(x4_up)
        x3_up = layers.Conv1D(256, kernel_size=2, padding='same', kernel_initializer='he_normal', activation='relu')(x3_up)
        # x3_up = layers.BatchNormalization()(x3_up)
        # x3_up = layers.ReLU()(x3_up)
        x3_up = x3 + x3_up
        x3_up = layers.Conv1D(256, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x3_up)
        # x3_up = layers.BatchNormalization()(x3_up)
        # x3_up = layers.ReLU()(x3_up)
        x3_up = layers.Conv1D(256, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x3_up)
        # x3_up = layers.BatchNormalization()(x3_up)
        # x3_up = layers.ReLU()(x3_up)

        x2_up = layers.UpSampling1D()(x3_up)
        x2_up = layers.Conv1D(128, kernel_size=2, padding='same', kernel_initializer='he_normal', activation='relu')(x2_up)
        # x2_up = layers.BatchNormalization()(x2_up)
        # x2_up = layers.ReLU()(x2_up)
        x2_up = x2 + x2_up
        x2_up = layers.Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x2_up)
        # x2_up = layers.BatchNormalization()(x2_up)
        # x2_up = layers.ReLU()(x2_up)
        x2_up = layers.Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x2_up)
        # x2_up = layers.BatchNormalization()(x2_up)
        # x2_up = layers.ReLU()(x2_up)

        x1_up = layers.UpSampling1D()(x2_up)
        x1_up = layers.Conv1D(64, kernel_size=2, padding='same', kernel_initializer='he_normal', activation='relu')(x1_up)
        # x1_up = layers.BatchNormalization()(x1_up)
        # x1_up = layers.ReLU()(x1_up)
        x1_up = x1 + x1_up
        x1_up = layers.Conv1D(64, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x1_up)
        # x1_up = layers.BatchNormalization()(x1_up)
        # x1_up = layers.ReLU()(x1_up)
        x1_up = layers.Conv1D(64, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x1_up)
        # x1_up = layers.BatchNormalization()(x1_up)
        # x1_up = layers.ReLU()(x1_up)
        x1_up = layers.Dropout(0.5)(x1_up)

        y = layers.Conv1D(2, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x1_up)
        y = layers.Conv1D(1, kernel_size=1, activation=self.final_act)(y)

        model = Model(inputs, y)
        return model

    # .................................................................................................................
    def conv_batchnorm_relu_block(self, input_tensor, nb_filter, kernel_size=3):

        x = layers.Conv1D(nb_filter, kernel_size, padding='same')(input_tensor)
        # x = layers.BatchNormalization(axis=2)(x)
        x = layers.Activation('relu')(x)

        return x

    def Conv1DTranspose(self, input_tensor, filters, kernel_size, strides=2, name=None, padding='same'):
        """
            input_tensor: tensor, with the shape (batch_size, time_steps, dims)
            filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
            kernel_size: int, size of the convolution kernel
            strides: int, convolution step size
            padding: 'same' | 'valid'
        """
        x = layers.Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
        x = layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), name=name, padding=padding)(x)
        x = layers.Lambda(lambda x: K.squeeze(x, axis=2))(x)
        return x

    def Unet2P(self, n_labels=1, using_deep_supervision=False):
        input_shape = (self.windows, 33)
        nb_filter = [64, 128, 256, 512, 1024]

        # Set image data format to channels first
        global bn_axis

        K.set_image_data_format("channels_last")
        bn_axis = -1
        inputs = layers.Input(shape=input_shape, name='input_image')

        conv1_1 = self.conv_batchnorm_relu_block(inputs, nb_filter=nb_filter[0])
        pool1 = layers.AvgPool1D(2, strides=2, name='pool1')(conv1_1)

        conv2_1 = self.conv_batchnorm_relu_block(pool1, nb_filter=nb_filter[1])
        pool2 = layers.AvgPool1D(2, strides=2, name='pool2')(conv2_1)

        up1_2 = self.Conv1DTranspose(conv2_1, nb_filter[0], 2, strides=2, name='up12', padding='same')
        conv1_2 = layers.concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
        conv1_2 = self.conv_batchnorm_relu_block(conv1_2, nb_filter=nb_filter[0])

        conv3_1 = self.conv_batchnorm_relu_block(pool2, nb_filter=nb_filter[2])
        pool3 = layers.AvgPool1D(2, strides=2, name='pool3')(conv3_1)

        up2_2 = self.Conv1DTranspose(conv3_1, nb_filter[1], 2, strides=2, name='up22', padding='same')
        conv2_2 = layers.concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
        conv2_2 = self.conv_batchnorm_relu_block(conv2_2, nb_filter=nb_filter[1])

        up1_3 = self.Conv1DTranspose(conv2_2, nb_filter[0], 2, strides=2, name='up13', padding='same')
        conv1_3 = layers.concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
        conv1_3 = self.conv_batchnorm_relu_block(conv1_3, nb_filter=nb_filter[0])

        conv4_1 = self.conv_batchnorm_relu_block(pool3, nb_filter=nb_filter[3])
        pool4 = layers.AvgPool1D(2, strides=2, name='pool4')(conv4_1)

        up3_2 = self.Conv1DTranspose(conv4_1, nb_filter[2], 2, strides=2, name='up32', padding='same')
        conv3_2 = layers.concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
        conv3_2 = self.conv_batchnorm_relu_block(conv3_2, nb_filter=nb_filter[2])

        up2_3 = self.Conv1DTranspose(conv3_2, nb_filter[1], 2, strides=2, name='up23', padding='same')
        conv2_3 = layers.concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
        conv2_3 = self.conv_batchnorm_relu_block(conv2_3, nb_filter=nb_filter[1])

        up1_4 = self.Conv1DTranspose(conv2_3, nb_filter[0], 2, strides=2, name='up14', padding='same')
        conv1_4 = layers.concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
        conv1_4 = self.conv_batchnorm_relu_block(conv1_4, nb_filter=nb_filter[0])

        conv5_1 = self.conv_batchnorm_relu_block(pool4, nb_filter=nb_filter[4])

        up4_2 = self.Conv1DTranspose(conv5_1, nb_filter[3], 2, strides=2, name='up42', padding='same')
        conv4_2 = layers.concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
        conv4_2 = self.conv_batchnorm_relu_block(conv4_2, nb_filter=nb_filter[3])

        up3_3 = self.Conv1DTranspose(conv4_2, nb_filter[2], 2, strides=2, name='up33', padding='same')
        conv3_3 = layers.concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
        conv3_3 = self.conv_batchnorm_relu_block(conv3_3, nb_filter=nb_filter[2])

        up2_4 = self.Conv1DTranspose(conv3_3, nb_filter[1], 2, strides=2, name='up24', padding='same')
        conv2_4 = layers.concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
        conv2_4 = self.conv_batchnorm_relu_block(conv2_4, nb_filter=nb_filter[1])

        up1_5 = self.Conv1DTranspose(conv2_4, nb_filter[0], 2, strides=2, name='up15', padding='same')
        conv1_5 = layers.concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
        conv1_5 = self.conv_batchnorm_relu_block(conv1_5, nb_filter=nb_filter[0])

        nestnet_output_1 = layers.Conv1D(n_labels, 1, activation='sigmoid', name='output_1', padding='same')(conv1_2)
        nestnet_output_2 = layers.Conv1D(n_labels, 1, activation='sigmoid', name='output_2', padding='same')(conv1_3)
        nestnet_output_3 = layers.Conv1D(n_labels, 1, activation='sigmoid', name='output_3', padding='same')(conv1_4)
        nestnet_output_4 = layers.Conv1D(n_labels, 1, activation='sigmoid', name='output_4', padding='same')(conv1_5)

        if using_deep_supervision:
            model = Model(input=inputs, output=[nestnet_output_1,
                                                nestnet_output_2,
                                                nestnet_output_3,
                                                nestnet_output_4])
        else:
            model = Model(inputs=inputs, outputs=nestnet_output_4)

        return model

    # .................................................................................................................
    def conv1d_block(self, inputs, filters, kernel_size=3, strides=1, padding='same'):
        Z = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
        # Z = layers.BatchNormalization(axis=-1)(Z)
        # A = keras.layers.PReLU(shared_axes=[1, 2])(Z)
        A = layers.ReLU()(Z)

        return A

    def Unet3P_conv1d(self, inputs, filters, n=2, kernel_size=3, stride=1, padding='same'):
        x = inputs

        for i in range(0, n+1):
            x = layers.Conv1D(filters, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False)(x)
            # x = layers.BatchNormalization()(x)
            # x = keras.layers.PReLU(shared_axes=[1, 2])(x)
            x = layers.ReLU()(x)

        return x

    def Unet3P(self, filters=[64, 128, 256, 512, 1024]):
        CatChannels = filters[0]
        CatBlocks = 5
        UpChannels = CatChannels * CatBlocks

        inputs = layers.Input((self.windows, 33))
        h1 = self.Unet3P_conv1d(inputs, filters[0])

        h2 = layers.MaxPooling1D()(h1)
        h2 = self.Unet3P_conv1d(h2, filters[1])

        h3 = layers.MaxPooling1D()(h2)
        h3 = self.Unet3P_conv1d(h3, filters[2])

        h4 = layers.MaxPooling1D()(h3)
        h4 = self.Unet3P_conv1d(h4, filters[3])

        h5 = layers.MaxPooling1D()(h4)
        hd5 = self.Unet3P_conv1d(h5, filters[4])

        # stage 4d
        h1_PT_hd4 = layers.MaxPool1D(strides=8)(h1)
        h1_PT_hd4 = self.conv1d_block(h1_PT_hd4, filters[0])

        h2_PT_hd4 = layers.MaxPool1D(strides=4)(h2)
        h2_PT_hd4 = self.conv1d_block(h2_PT_hd4, filters[1])

        h3_PT_hd4 = layers.MaxPool1D(strides=2)(h3)
        h3_PT_hd4 = self.conv1d_block(h3_PT_hd4, filters[2])

        h4_Cat_hd4 = self.conv1d_block(h4, filters[3])

        hd5_UT_hd4 = layers.UpSampling1D()(hd5)
        hd5_UT_hd4 = self.conv1d_block(hd5_UT_hd4, filters[4])

        hd4 = layers.concatenate([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4])
        hd4 = self.conv1d_block(hd4, UpChannels)

        # stage 3d
        h1_PT_hd3 = layers.MaxPool1D(strides=4)(h1)
        h1_PT_hd3 = self.conv1d_block(h1_PT_hd3, filters[0])

        h2_PT_hd3 = layers.MaxPool1D(strides=2)(h2)
        h2_PT_hd3 = self.conv1d_block(h2_PT_hd3, filters[1])

        h3_Cat_hd3 = self.conv1d_block(h3, filters[2])

        hd4_UT_hd3 = layers.UpSampling1D()(hd4)
        hd4_UT_hd3 = self.conv1d_block(hd4_UT_hd3, UpChannels)

        hd5_UT_hd3 = layers.UpSampling1D(size=4)(hd5)
        hd5_UT_hd3 = self.conv1d_block(hd5_UT_hd3, UpChannels)

        hd3 = layers.concatenate([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3])
        hd3 = self.conv1d_block(hd3, UpChannels)

        # stage 2d
        h1_PT_hd2 = layers.MaxPool1D(strides=2)(h1)
        h1_PT_hd2 = self.conv1d_block(h1_PT_hd2, filters[0])

        h2_Cat_hd2 = self.conv1d_block(h2, filters[1])

        hd3_UT_hd2 = layers.UpSampling1D()(hd3)
        hd3_UT_hd2 = self.conv1d_block(hd3_UT_hd2, UpChannels)

        hd4_UT_hd2 = layers.UpSampling1D(size=4)(hd4)
        hd4_UT_hd2 = self.conv1d_block(hd4_UT_hd2, UpChannels)

        hd5_UT_hd2 = layers.UpSampling1D(size=8)(hd5)
        hd5_UT_hd2 = self.conv1d_block(hd5_UT_hd2, UpChannels)

        hd2 = layers.concatenate([h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2])
        hd2 = self.conv1d_block(hd2, UpChannels)

        # stage 1d
        h1_Cat_hd1 = self.conv1d_block(h1, filters[0])

        hd2_UT_hd1 = layers.UpSampling1D()(hd2)
        hd2_UT_hd1 = self.conv1d_block(hd2_UT_hd1, UpChannels)

        hd3_UT_hd1 = layers.UpSampling1D(size=4)(hd3)
        hd3_UT_hd1 = self.conv1d_block(hd3_UT_hd1, UpChannels)

        hd4_UT_hd1 = layers.UpSampling1D(size=8)(hd4)
        hd4_UT_hd1 = self.conv1d_block(hd4_UT_hd1, UpChannels)

        hd5_UT_hd1 = layers.UpSampling1D(size=16)(hd5)
        hd5_UT_hd1 = self.conv1d_block(hd5_UT_hd1, UpChannels)

        hd1 = layers.concatenate([h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1])
        hd1 = self.conv1d_block(hd1, UpChannels)

        d5 = layers.Conv1D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hd5)
        d5 = layers.UpSampling1D(size=16)(d5)
        d5 = layers.Activation(self.final_act, name='d5')(d5)

        d4 = layers.Conv1D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hd4)
        d4 = layers.UpSampling1D(size=8)(d4)
        d4 = layers.Activation(self.final_act, name='d4')(d4)

        d3 = layers.Conv1D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hd3)
        d3 = layers.UpSampling1D(size=4)(d3)
        d3 = layers.Activation(self.final_act, name='d3')(d3)

        d2 = layers.Conv1D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hd2)
        d2 = layers.UpSampling1D()(d2)
        d2 = layers.Activation(self.final_act, name='d2')(d2)

        d1 = layers.Conv1D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hd1)

        d1 = layers.Activation(self.final_act, name='d1')(d1)
        d = layers.average([d1, d2, d3, d4, d5])
        d = layers.Conv1D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d)
        d = layers.Activation(self.final_act, name='d')(d)

        model = Model(inputs=inputs, outputs=d)
        return model

    # .................................................................................................................

    def Recurrent_block(self, x, filters, t=2):
        for i in range(t):
            if i == 0:
                x1 = layers.Conv1D(filters, 3, strides=1, padding='same', activation='relu', use_bias=True)(x)
            x1 = layers.Conv1D(filters, 3, strides=1, padding='same', activation='relu', use_bias=True)(x + x1)

        return x1

    def Res_block(self, x, filters):
        x = layers.Conv1D(filters, 1, strides=1, padding='same', activation='relu', use_bias=True)(x)

        x1 = self.Recurrent_block2P(x, filters=filters, t=2)
        x1 = self.Recurrent_block2P(x1, filters=filters, t=2)

        return x + x1

    def RUnet(self):
        inputs = layers.Input((self.windows, 33))
        x1 = layers.Conv1D(self.filter[0], kernel_size=1, padding='same', activation='relu')(inputs)
        x1_ = layers.Conv1D(self.filter[0], kernel_size=3, padding='same', activation='relu')(x1)
        x1_ = layers.Conv1D(self.filter[0], kernel_size=3, padding='same', activation='relu')(x1 + x1_)
        x1 = x1 + x1_

        x2 = layers.MaxPooling1D()(x1)
        x2 = layers.Conv1D(self.filter[1], kernel_size=1, padding='same', activation='relu')(x2)
        x2_ = layers.Conv1D(self.filter[1], kernel_size=3, padding='same', activation='relu')(x2)
        x2_ = layers.Conv1D(self.filter[1], kernel_size=3, padding='same', activation='relu')(x2 + x2_)
        x2 = x2 + x2_

        x3 = layers.MaxPooling1D()(x2)
        x3 = layers.Conv1D(self.filter[2], kernel_size=1, padding='same', activation='relu')(x3)
        x3_ = layers.Conv1D(self.filter[2], kernel_size=3, padding='same', activation='relu')(x3)
        x3_ = layers.Conv1D(self.filter[2], kernel_size=3, padding='same', activation='relu')(x3 + x3_)
        x3 = x3 + x3_

        x4 = layers.MaxPooling1D()(x3)
        x4 = layers.Conv1D(self.filter[3], kernel_size=1, padding='same', activation='relu')(x4)
        x4_ = layers.Conv1D(self.filter[3], kernel_size=3, padding='same', activation='relu')(x4)
        x4_ = layers.Conv1D(self.filter[3], kernel_size=3, padding='same', activation='relu')(x4 + x4_)
        x4 = x4 + x4_

        x5 = layers.MaxPooling1D()(x4)
        x5 = layers.Conv1D(self.filter[4], kernel_size=1, padding='same', activation='relu')(x5)
        x5_ = layers.Conv1D(self.filter[4], kernel_size=3, padding='same', activation='relu')(x5)
        x5_ = layers.Conv1D(self.filter[4], kernel_size=3, padding='same', activation='relu')(
            x5 + x5_)
        x5 = x5 + x5_

        x4_up = layers.UpSampling1D()(x5)
        x4_up = layers.Conv1D(self.filter[3], kernel_size=3, padding='same', activation='relu')(x4_up)
        x4_up = x4_up + x4
        x4_up = layers.Conv1D(self.filter[3], kernel_size=1, padding='same', activation='relu')(x4_up)
        x4_up_ = layers.Conv1D(self.filter[3], kernel_size=3, padding='same', activation='relu')(x4_up)
        x4_up_ = layers.Conv1D(self.filter[3], kernel_size=3, padding='same', activation='relu')(
            x4_up + x4_up_)
        x4_up = x4_up + x4_up_

        x3_up = layers.UpSampling1D()(x4_up)
        x3_up = layers.Conv1D(self.filter[2], kernel_size=3, padding='same', activation='relu')(x3_up)
        x3_up = x3_up + x3
        x3_up = layers.Conv1D(self.filter[2], kernel_size=1, padding='same', activation='relu')(x3_up)
        x3_up_ = layers.Conv1D(self.filter[2], kernel_size=3, padding='same', activation='relu')(x3_up)
        x3_up_ = layers.Conv1D(self.filter[2], kernel_size=3, padding='same', activation='relu')(
            x3_up + x3_up_)
        x3_up = x3_up + x3_up_

        x2_up = layers.UpSampling1D()(x3_up)
        x2_up = layers.Conv1D(self.filter[1], kernel_size=3, padding='same', activation='relu')(x2_up)
        x2_up = x2_up + x2
        x2_up = layers.Conv1D(self.filter[1], kernel_size=1, padding='same', activation='relu')(x2_up)
        x2_up_ = layers.Conv1D(self.filter[1], kernel_size=3, padding='same', activation='relu')(x2_up)
        x2_up_ = layers.Conv1D(self.filter[1], kernel_size=3, padding='same', activation='relu')(
            x2_up + x2_up_)
        x2_up = x2_up + x2_up_

        x1_up = layers.UpSampling1D()(x2_up)
        x1_up = layers.Conv1D(self.filter[0], kernel_size=2, padding='same', activation='relu')(x1_up)
        x1_up = x1_up + x1
        x1_up = layers.Conv1D(self.filter[0], kernel_size=1, padding='same', activation='relu')(x1_up)
        x1_up_ = layers.Conv1D(self.filter[0], kernel_size=3, padding='same', activation='relu')(x1_up)
        x1_up_ = layers.Conv1D(self.filter[0], kernel_size=3, padding='same', activation='relu')(
            x1_up + x1_up_)
        x1_up = x1_up + x1_up_
        x1_up = layers.Dropout(0.5)(x1_up)

        y = layers.Conv1D(1, kernel_size=1, activation=self.final_act)(x1_up)

        model = Model(inputs, y)
        return model

    # .................................................................................................................
    def conv1d_block_Att(self, inputs, filters, kernel_size=3, strides=1, padding='same'):
        Z = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, activation='relu')(inputs)
        # Z = layers.BatchNormalization(axis=-1)(Z)
        # A = keras.layers.PReLU(shared_axes=[1, 2])(Z)
        Z = layers.ReLU()(Z)

        A = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, activation='relu')(Z)
        # A = layers.BatchNormalization(axis=-1)(A)
        # A = keras.layers.PReLU(shared_axes=[1, 2])(Z)
        A = layers.ReLU()(A)

        return A

    def up_conv_Att(self, inputs, filters):
        x = layers.UpSampling1D()(inputs)
        x = layers.Conv1D(filters, 3, strides=1, padding='same', use_bias=True)(x)
        # x = layers.BatchNormalization(axis=-1)(x)
        x = layers.ReLU()(x)

        return x

    def Att_block(self, g, x, filters):
        g = layers.Conv1D(filters, 1, strides=1, padding='same', use_bias=True)(g)
        # g1 = layers.BatchNormalization()(g)

        x = layers.Conv1D(filters, 2, strides=1, padding='same', use_bias=True)(x)
        # x1 = layers.BatchNormalization()(x)

        # psi = layers.ReLU()(g1 + x1)
        psi = layers.ReLU()(g + x)
        psi = layers.Conv1D(1, 1, strides=1, padding='same', use_bias=True, activation='sigmoid')(psi)
        # psi = layers.BatchNormalization()(psi)

        return x*psi

    def AttUnet(self):
        inputs = layers.Input((self.windows, 33))
        x1 = self.conv1d_block_Att(inputs, 64, kernel_size=3, padding='same')

        x2 = layers.MaxPooling1D()(x1)
        x2 = self.conv1d_block_Att(x2, 128, kernel_size=3, padding='same')

        x3 = layers.MaxPooling1D()(x2)
        x3 = self.conv1d_block_Att(x3, 256, kernel_size=3, padding='same')

        x4 = layers.MaxPooling1D()(x3)
        x4 = self.conv1d_block_Att(x4, 512, kernel_size=3, padding='same')

        x5 = layers.MaxPooling1D()(x4)
        x5 = self.conv1d_block_Att(x5, 1024, kernel_size=3, padding='same')
        x5 = layers.Dropout(0.5)(x5)

        x4_up = self.up_conv_Att(x5, 512)
        x4 = self.Att_block(x4_up, x4, 512)
        x4_up = x4_up + x4
        x4_up = self.conv1d_block_Att(x4_up, 512, kernel_size=3, padding='same')

        x3_up = self.up_conv_Att(x4_up, 256)
        x3 = self.Att_block(x3_up, x3, 256)
        x3_up = x3_up + x3
        x3_up = self.conv1d_block_Att(x3_up, 256, kernel_size=3, padding='same')

        x2_up = self.up_conv_Att(x3_up, 128)
        x2 = self.Att_block(x2_up, x2, 128)
        x2_up = x2_up + x2
        x2_up = self.conv1d_block_Att(x2_up, 128, kernel_size=3, padding='same')

        x1_up = self.up_conv_Att(x2_up, 64)
        x1 = self.Att_block(x1_up, x1, 64)
        x1_up = x1_up + x1
        x1_up = self.conv1d_block_Att(x1_up, 64, kernel_size=3, padding='same')
        x1_up = layers.Dropout(0.5)(x1_up)

        y = layers.Conv1D(1, kernel_size=1, activation=self.final_act)(x1_up)

        model = Model(inputs, y)
        return model

if __name__ == '__main__':
    model = Unet_model().AttUnet()
    model.summary()