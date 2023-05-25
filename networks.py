'''
What Characterizes Personalities of Graphic Designs? SIGGRAPH 2018
Network Architecture
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import h5py
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Convolution2D, Flatten, Activation, Dropout, ZeroPadding2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Merge
from keras import backend as K
from keras.regularizers import l2, activity_l2


def create_design_feature_network(img_chns,output_dim,img_width,img_height,dropout_rate,w_regularizer,batch_norm_flag):
    '''
     Design feature network
    :param img_chns:
    :param output_dim:
    :param img_width:
    :param img_height:
    :param dropout_rate:
    :param w_regularizer:
    :return:
    '''
    ##########################################
    model_feature = Sequential()
    if dropout_rate[0]!=-1:
        model_feature.add(Dropout(dropout_rate[0],input_shape=(img_chns, img_width, img_height),name='feature_dropout0'))
        model_feature.add(Convolution2D(64, 3, 3, border_mode='same',
                                        activation='relu', W_regularizer=l2(w_regularizer), name='feature_conv1_1'))
    else:
        model_feature.add(Convolution2D(64, 3, 3,input_shape=(img_chns, img_width, img_height), border_mode='same',
                                    activation='relu',W_regularizer=l2(w_regularizer), name='feature_conv1_1'))
    model_feature.add(MaxPooling2D((4, 4),name='feature_maxpool1_2'))
    if dropout_rate[2]!=-1:
        model_feature.add(Dropout(dropout_rate[2], name='feature_dropout2'))
    if batch_norm_flag==True:
        model_feature.add(BatchNormalization(mode=2, axis=1,name='feature_batchnorm1'))

    ##########################################
    model_feature.add(Convolution2D(64, 3, 3, border_mode='same',activation='relu', W_regularizer=l2(w_regularizer),
                                    name='feature_conv2_1'))
    model_feature.add(MaxPooling2D((4, 4), name='feature_maxpool2'))
    if dropout_rate[4]!=-1:
        model_feature.add(Dropout(dropout_rate[4], name='feature_dropout2_2'))
    if batch_norm_flag==True:
        model_feature.add(BatchNormalization(mode=2, axis=1,name='feature_batch2'))

    ##########################################
    model_feature.add(Convolution2D(64, 3, 3, border_mode='same',activation='relu', W_regularizer=l2(w_regularizer),
                                    name='feature_conv3_1'))
    model_feature.add(MaxPooling2D((4, 4), name='feature_maxpool3'))
    if dropout_rate[6]!=-1:
        model_feature.add(Dropout(dropout_rate[6], name='feature_dropout3_2'))
    if batch_norm_flag==True:
        model_feature.add(BatchNormalization(mode=2, axis=1,name='feature_batch3'))

    ##########################################
    model_feature.add(Convolution2D(64, 3, 3 , border_mode='same', activation='relu',
                                    W_regularizer=l2(w_regularizer),name='feature_conv4_1'))
    if dropout_rate[7]!=-1:
        model_feature.add(Dropout(dropout_rate[7], name='feature_dropout4_1'))

    ##########################################
    model_feature.add(Flatten(name='feature_flatten'))
    model_feature.add(Dense(output_dim, activation='relu', W_regularizer=l2(w_regularizer),name='feature_dense1'))
    if dropout_rate[8]!=-1:
        model_feature.add(Dropout(dropout_rate[8], name='feature_dropout5'))
    return model_feature

def create_semantic_embedding_network(input_dim,intermediate_dim,dropout_rate,w_regularizer):
    '''
    Semantic embedding network
    :param input_dim:
    :param intermediate_dim:
    :param dropout_rate:
    :param w_regularizer:
    :return:
    '''
    mlp_model = Sequential()
    mlp_model.add(Dense(intermediate_dim[0], input_shape=(input_dim,), W_regularizer=l2(w_regularizer),name='semantic_dense1'))
    mlp_model.add(Activation('relu',name='semantic_activation1'))
    mlp_model.add(Dense(intermediate_dim[1], W_regularizer=l2(w_regularizer),name='semantic_dense2'))
    mlp_model.add(Activation('relu',name='semantic_activation2'))
    return mlp_model

def create_base_network_merge(model1,model2):
    merged_model = Sequential()
    merged_model.add(Merge([model1, model2], mode='concat', concat_axis=1))
    return merged_model


def create_semantic_scoring_network(input_dim,intermediate_dim,dropout_rate,w_regularizer):
    '''
    Semantic scoring network
    :param input_dim:
    :param intermediate_dim:
    :param dropout_rate:
    :param w_regularizer:
    :return:
    '''
    ##########################################
    mlp_model = Sequential()
    mlp_model.add(Dense(intermediate_dim[0], input_shape=(input_dim,), W_regularizer=l2(w_regularizer),name='mlp_dense1'))
    mlp_model.add(Activation('relu',name='mlp_activation1'))
    if dropout_rate[0]!=-1:
        mlp_model.add(Dropout(dropout_rate[0], name='mlp_dropout1'))

    ##########################################
    mlp_model.add(Dense(intermediate_dim[1], W_regularizer=l2(w_regularizer),name='mlp_dense2'))
    mlp_model.add(Activation('relu',name='mlp_activation2'))
    if dropout_rate[1]!=-1:
        mlp_model.add(Dropout(dropout_rate[1], name='mlp_dropout2'))

    ##########################################
    mlp_model.add(Dense(1, W_regularizer=l2(w_regularizer),name='mlp_dense3'))
    return mlp_model
