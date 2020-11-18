import os
#import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from models.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import backend

class Model(BaseModel):
    def __init__(self, history, params=None):
        self.name = 'model5'
        self.model = Sequential()

        # 64x64
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(64,64,1)))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))

        # 32x32
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))

        # 16x16
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))

        # 8x8
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))

        # 4x4
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))

        # flatten
        self.model.add(Flatten())

        # Adding Dropout after flatten in user intends to
        if params is not None:
            if 'dropout' in params:
                self.model.add(Dropout(params['dropout']))

        self.model.add(Dense(1, activation=tf.nn.sigmoid))

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(params)