import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten,MaxPooling2D,BatchNormalization,Lambda, AveragePooling2D
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import pandas as pd
import random

X_data = np.loadtxt('new_train_features.csv',skiprows=1,delimiter=',')
Y_data = np.loadtxt('new_train_target.csv',skiprows=1,delimiter=',')
X_predict = np.loadtxt('test_features.csv',skiprows=1,delimiter=',')

def reshape(data, X_or_Y):
    if(X_or_Y == 'X'):
        data = data[:, 1:]
        data = data.reshape(int(len(data)/375),375,5,1)
    elif(X_or_Y == 'Y'):
        data = data[:, 1:3]
    print(data.shape)

    return data

def train_test_split(X_data, Y_data):
    test_set_count = 3  # total_set_count = 35
    set_size = 80  # (-400,-400) ... (400,400)

    X_test = X_data[:test_set_count * set_size]
    X_train = X_data[test_set_count * set_size:]
    print('X_train_shape: {}'.format(X_train.shape))
    print('X_test_shape: {}'.format(X_test.shape))

    Y_test = Y_data[:test_set_count * set_size]
    Y_train = Y_data[test_set_count * set_size:]
    print('Y_train_shape: {}'.format(Y_train.shape))
    print('Y_test_shape: {}'.format(Y_test.shape))

    return X_train, X_test, Y_train, Y_test

def my_loss(y_true, y_pred):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])

    return K.mean(K.square(divResult))

def my_loss_E1(y_true, y_pred):
    #weight = np.array([1,1])
    return K.mean(K.square(y_true-y_pred))/2e+04


def set_model():
    padding = 'valid'
    activation = 'elu'
    model = Sequential()
    filters = 16
    kernel_size = (3, 1)

    model.add(Conv2D(filters, kernel_size, padding=padding, activation=activation, input_shape=(375, 5, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(filters * 2, kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(filters * 4, kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(filters * 8, kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(filters * 16, kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(filters * 32, kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(2))

    optimizer = keras.optimizers.Adam()
    model.compile(loss=my_loss_E1,
                  optimizer=optimizer,
                  )
    model.summary()

    return model

def train(model, X, Y):
    best_save = ModelCheckpoint('best_m.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    set_size = 80
    total_set_count = int(len(X)/set_size)
    val_set_count = int(total_set_count*0.2)
    epoch_size = 100

    X = X.reshape(total_set_count,set_size,375,5,1)
    Y = Y.reshape(total_set_count,set_size,2)
    print('X: {}'.format(X.shape))
    print('Y: {}'.format(Y.shape))

    for epochs in range(epoch_size):
        X_val = list()
        X_train = list()
        Y_val = list()
        Y_train = list()

        set_num = list(range(total_set_count))
        random.shuffle(set_num)
        for index, value in enumerate(set_num):
            if(index < val_set_count):
                X_val.append(X[value])
                Y_val.append(Y[value])
            else:
                X_train.append(X[value])
                Y_train.append(Y[value])

        X_val = np.array(X_val)
        X_train = np.array(X_train)
        Y_val = np.array(Y_val)
        Y_train = np.array(Y_train)
        # print('X_train_shape: {}'.format(X_train.shape))
        # print('X_val_shape: {}'.format(X_val.shape))
        # print('Y_train_shape: {}'.format(Y_train.shape))
        # print('Y_val_shape: {}'.format(Y_val.shape))

        X_val = X_val.reshape(len(X_val) * set_size, 375, 5, 1)
        X_train = X_train.reshape(len(X_train) * set_size, 375, 5, 1)
        Y_val = Y_val.reshape(len(Y_val) * set_size, 2)
        Y_train = Y_train.reshape(len(Y_train) * set_size, 2)
        # print('X_train_shape: {}'.format(X_train.shape))
        # print('X_val_shape: {}'.format(X_val.shape))
        # print('Y_train_shape: {}'.format(Y_train.shape))
        # print('Y_val_shape: {}'.format(Y_val.shape))
        print('epoch: {}'.format(epochs))
        history = model.fit(X_train, Y_train,
                            epochs=1,
                            batch_size=80,
                            validation_data=(X_val, Y_val),
                            verbose=2,
                            callbacks=[best_save])

    # fig, loss_ax = plt.subplots()
    # loss_ax.plot(history.history['loss'], 'y', label='train loss')
    # loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    # loss_ax.set_xlabel('epoch')
    # loss_ax.set_ylabel('loss')
    # loss_ax.legend(loc='upper left')
    # plt.show()
    #plt.savefig('loss_XY.png')

    return model


def load_best_model():
    model = load_model('best_m.hdf5', custom_objects={'my_loss_E1': my_loss, })
    return model


submit = pd.read_csv('sample_submission.csv')
X_data = reshape(X_data, 'X')
Y_data = reshape(Y_data, 'Y')
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data)

model = set_model()
train(model, X_train, Y_train)
# best_model = load_best_model()

test_loss = model.evaluate(X_test, Y_test)
# test_loss = best_model.evaluate(X_test, Y_test)
print('test_loss: ', test_loss)

X_predict = reshape(X_predict, 'X')
Y_predict = model.predict(X_predict)

submit.iloc[:, 1] = Y_predict[:, 0]
submit.iloc[:, 2] = Y_predict[:, 1]

submit.to_csv('result/submit.csv', index = False)