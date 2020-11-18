import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import bz2
import tensorflow

# import models
import models.model1 as model1
import models.model2 as model2
import models.model3 as model3
import models.model4 as model4
import models.model5 as model5
import models.model6 as model6
import models.model7 as model7

import gc
import re
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# user options
model_name = 'model1'
use_adam = True
learning_rates = [0.000001]
decay_rate = 0
decay_epochs = 50
momentum = 0.9
batch_sizes = [16]
epochs = 300
plot = True
train = False
predict = True

if model_name == 'model7':
    use_data_size = 256; # possible options [64, 128, 256]
else:
    use_data_size = 64

params = {}
params['resetHistory']  = False
params['print_summary'] = True
params['dropout'] = 0.5
params['data_aug_enable'] = False
params['models_dir'] = '../trained_models/' + model_name
#params['models_dir'] = '../trained_models/model3_hv'

if model_name == 'model7':
    params['poisson'] = True
else:
    params['poisson'] = False

# data files
data_in_dir = "../data"
train_data_file = data_in_dir + "/train_" + str(use_data_size) + "_data.bz2"
dev_data_file = data_in_dir + "/dev_" + str(use_data_size) + "_data.bz2"
#dev_data_file = data_in_dir + "/test_" + str(use_data_size) + "_data.bz2"
test_data_file = data_in_dir + "/test_" + str(use_data_size) + "_data.bz2"

def load_data_from_file(fname, dname):
    if re.match(".*.bz2", fname):
        fin = bz2.BZ2File(fname, 'rb')
        try:
            print("Reading data from file %s" % (fname))
            data0, data1, data2 = pickle.load(fin)
        finally:
            fin.close()
    else:
        print("Reading data from file %s" % (fname))
        with open(fname, 'rb') as fin:
            data0, data1, data2  = pickle.load(fin)

    print ("%s shape: %s" %(dname, data0.shape))
    print("%s shape: %s" %(dname, data1.shape))
    print("%s shape: %s" % (dname, data2.shape))
    return data0, data1, data2


# LossHistory Class
class LossHistory(tensorflow.keras.callbacks.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []
        self.acc_epochs = 0
        super(LossHistory, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if 'poisson' in params and params['poisson']:
            self.train_losses.append(logs.get('loss'))
            self.train_acc.append(logs.get('mean_squared_error'))
            self.val_losses.append(logs.get('val_loss'))
            self.val_acc.append(logs.get('val_mean_squared_error'))
        else:
            self.train_losses.append(logs.get('loss'))
            self.train_acc.append(logs.get('accuracy'))
            self.val_losses.append(logs.get('val_loss'))
            self.val_acc.append(logs.get('val_accuracy'))
        gc.collect()
        if epoch%5 == 0:
            # Save model
            print ("Saving the model in ../experiments/current/m_" + str(epoch))
            model.save('../experiments/current/m_' + str(epoch))

# learning rate scheduler
def lr_scheduler(epoch, lr):
    global decay_rate, decay_epochs
    if epoch%decay_epochs == 0 and epoch and decay_rate != 0:
        return lr * decay_rate
    return lr

## Define the model
x_train, y_train, c_train = load_data_from_file(train_data_file, "train")
x_dev, y_dev, c_dev = load_data_from_file(dev_data_file, "dev")

if 'poisson' in params and params['poisson']:
    y_train = c_train
    y_dev = c_dev

# reshape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_dev = np.reshape(x_dev, (x_dev.shape[0], x_dev.shape[1], x_dev.shape[2], 1))

for batch_size in batch_sizes:
    for lr in learning_rates:
        if train or 'models_dir' not in params:
            params['models_dir'] = f'../experiments/{model_name}/{batch_size}.{lr}'
        history = LossHistory()
        if model_name == 'model1':
            model = model1.Model(history, params)
        elif model_name == 'model2':
            model = model2.Model(history, params)
        elif model_name == 'model3':
            model = model3.Model(history, params)
        elif model_name == 'model4':
            model = model4.Model(history, params)
        elif model_name == 'model5':
            model = model5.Model(history, params)
        elif model_name == 'model6':
            model = model6.Model(history, params)
        elif model_name == 'model7':
            model = model7.Model(history, params)
        else:
            model = None
            exit("Something went wrong, model not defined")

        if not train:
            model.is_train = False

        ## training
        if use_adam:
            optimizer = tensorflow.keras.optimizers.Adam(lr=lr)
        else:
            optimizer = tensorflow.keras.optimizers.SGD(lr=lr, momentum=momentum)

        model.compile(optimizer)

        #Load data into model
        if train:
            model.x_train = x_train
            model.y_train = y_train
            print(model.x_train.dtype)

        model.x_val = x_dev
        model.y_val = y_dev
        print (model.x_val.dtype)

        # instantiate model
        if train:
            model.train(batch_size, epochs, lr_scheduler)
            model.save()

        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            model.train_plot(fig, ax, show_plot=False)

        if predict:
            y_hat = model.my_predict(x_dev, batch_size)
            if 'poisson' in params and params['poisson']:
                n = y_dev.shape[0]
                mse = np.square(y_hat-y_dev.reshape(n, 1)).mean(axis=None)
                print("Average MSE on dev set is %.2f" % (mse))
                print (y_dev.shape, y_hat.shape)
                print (np.max(y_dev))
                print(np.max(y_hat))
                plt.figure()
                plt.plot(y_dev, y_hat, 'bx', Linewidth=2)
                plt.xlabel('real count')
                plt.ylabel('predicted count')
            else:
                y_pred = y_hat > model.params['y_hat_threshold']
                n = y_dev.shape[0]
                y_err = y_pred != y_dev.reshape(n, 1)
                acc = (n - np.sum(y_err)) * 100 / n
                print (y_pred)
                print("Accuracy on dev set is %.2f" % (acc))


if plot or train:
    plt.show()

