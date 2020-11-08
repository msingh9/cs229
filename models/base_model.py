# Libraries
import os
import pickle
import tensorflow as tf
from matplotlib.patches import Rectangle
import random
from tensorflow.keras import backend

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing. image import ImageDataGenerator
import shutil

from tensorflow.keras.models import Model as m
from matplotlib import pyplot
from tensorflow.keras.layers import Input, Reshape, UpSampling2D, Lambda, dot
import bz2
from PIL import Image

random.seed(1223143)

# base class for all models
# Describe each method (function)
class BaseModel:
    def __init__(self, params=None):
        self.params = {}
        # One line description for each variable
        self.params['resetHistory'] = False
        self.params['models_dir'] = "."
        self.params['print_summary'] = True
        self.params['y_hat_threshold'] = 0.5

        self.patience = 8
        self.is_train = True
        self.all_plot = False

        # data placeholders
        self.x_train = None
        self.y_train = None

        self.x_val = None
        self.y_val = None

        self.x_test = None
        self.y_test = None

        # setting parameters based on the params as given by the user
        if params is not None:
            for key, value in params.items():
                self.params[key] = value

        # combining model directory and name of model
        self.name = self.params['models_dir'] + '/' + self.name

        # create model directory if unknown
        if not os.path.isdir(self.params['models_dir']):
            os.makedirs(self.params['models_dir'])

        # Copy the train.py
        if self.is_train:
            shutil.copyfile('train.py', self.params['models_dir'] + '/train.py.copy')

        # load model if model is already there
        print (self.name)
        if not self.params['resetHistory'] and os.path.isfile(self.name + '.h5'):
            print("Loading model from " + self.name + '.h5')
            self.model = load_model(self.name + '.h5')
            if self.history:
                with open(self.name + '.aux_data', 'rb') as fin:
                    self.history.train_losses, self.history.val_losses, self.history.train_acc, self.history.val_acc = pickle.load(fin)

        # Check if model is defined
        if not self.model:
            exit("model not defined")

        # print summary if user intends to
        if self.params['print_summary']:
            print(self.model.summary())

    def save(self, name=None):
        self.sfname = self.name
        if name:
            self.sfname = name
        self.model.save(self.sfname + '.h5')
        with open(self.sfname + '.aux_data', 'wb') as fout:
            pickle.dump((self.history.train_losses, self.history.val_losses, self.history.train_acc, self.history.val_acc), fout)
        if not name:
            plot_model(self.model, to_file=self.name + '.png')

    def compile(self, optimizer):
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, batch_size, epochs, lr_scheduler):
        if self.params['data_aug_enable']:
            if self.x_train.dtype == np.uint8:
                norm_const = np.array(255).astype('float16')
                self.x_train = self.x_train / norm_const

            if self.x_val.dtype == np.uint8:
                norm_const = np.array(255).astype('float16')
                self.x_val = self.x_val / norm_const

            self.datagen = ImageDataGenerator(rotation_range = 40, width_shift_range=0.3, height_shift_range=0.3)
            self.model.fit_generator(self.datagen.flow(self.x_train, self.y_train, batch_size=batch_size),
                                     steps_per_epoch=len(self.x_train)/batch_size,
                                     epochs=epochs, validation_data=(self.x_val, self.y_val),
                                     callbacks=[self.history,
                                                tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
                                                tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=self.patience)])
            return

        # default
        if self.x_train.dtype == np.uint8:
            norm_const = np.array(255).astype('float16')
            self.x_train = self.x_train / norm_const

        if self.x_val.dtype == np.uint8:
            norm_const = np.array(255).astype('float16')
            self.x_val = self.x_val / norm_const

        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(self.x_val, self.y_val),
                           callbacks = [self.history,
                                        tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
                                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)])

    def train_plot(self, fig=None, ax=None, show_plot=True, label=None):
        if not label:
            label = self.name
        if not fig:
            fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(self.history.train_losses[self.history.acc_epochs:], label=label + ' train', color='red')
        ax[0].plot(self.history.val_losses[self.history.acc_epochs:], label=label +' val', color='blue')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('epocs')
        ax[0].set_title("Loss vs epocs, train(Red)")

        ax[1].plot(self.history.train_acc[self.history.acc_epochs:], label=label + ' train', color='red')
        ax[1].plot(self.history.val_acc[self.history.acc_epochs:], label=label + ' val', color='blue')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('epocs')
        ax[1].set_title("Accuracy vs epocs, train(Red)")

        print('train_loss: ' + str(self.history.train_losses[-5:-1]))
        print('val_loss: ' + str(self.history.val_losses[-5:-1]))
        print('train_acc: ' + str(self.history.train_acc[-5:-1]))
        print('val_acc: ' + str(self.history.val_acc[-5:-1]))
        print('epochs:   ' + str(len(self.history.train_losses)))

        if show_plot:
            plt.show()

    # over write predict method
    def predict(self, x, batchsize):
        return self.model.predict(x, batchsize)
