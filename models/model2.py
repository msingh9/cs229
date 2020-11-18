import os
#import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Sequential
from models.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.models import Model as m
from tensorflow.keras import backend

class Model(BaseModel):
    def __init__(self, history, params=None):
        self.name = 'model2'
        self.model = Sequential()

        inputs = Input(shape=(64, 64, 1))

        # 64x64
        conv_7x7 = Conv2D(64, (7, 7), activation='relu')(inputs)
        conv_5x5 = Conv2D(64, (5, 5), activation='relu')(inputs)
        conv_3x3 = Conv2D(64, (3, 3), activation='relu')(inputs)
        maxpool_7x7 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv_7x7)
        maxpool_5x5 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv_5x5)
        maxpool_3x3 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv_3x3)

        # flatten
        flatten_7x7 = Flatten()(maxpool_7x7)
        flatten_5x5 = Flatten()(maxpool_5x5)
        flatten_3x3 = Flatten()(maxpool_3x3)

        # concatenate
        flatten = concatenate([flatten_7x7, flatten_5x5, flatten_3x3])

        # Adding Dropout after flatten in user intends to
        if params is not None:
            if 'dropout' in params:
                dropout = Dropout(params['dropout'])(flatten)

        dense = Dense(1, activation=tf.nn.sigmoid)(dropout)

        # Model
        self.model = m(inputs=[inputs], outputs=dense)

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(params)