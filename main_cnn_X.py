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
from tensorflow.python.keras.layers import Dropout
from _datetime import datetime

X_data = np.loadtxt('new_train_features.csv',skiprows=1,delimiter=',')
Y_data = np.loadtxt('new_train_target.csv',skiprows=1,delimiter=',')
X_predict = np.loadtxt('test_features.csv',skiprows=1,delimiter=',')

def reshape(data, X_or_Y):
    if(X_or_Y == 'X'):
        data = data[:, 1:]
        data = data.reshape(int(len(data)/375),375,5,1)
        print('reshaped X: {}'.format(data.shape))
    elif(X_or_Y == 'Y'):
        data = data[:, 1:3]
        print('reshaped Y: {}'.format(data.shape))

    return data

def augmentation(X_data, Y_data, fold):
    X_output = X_data.tolist()
    Y_output = Y_data.tolist()
    for iteration in range(fold-1):
        multiple = random.uniform(0.5, 1.5)
        for data in X_data:
            new_data = list()
            for sequence in data:
                new_sequence = list()
                for index in range(len(sequence)):
                    if index == 0:
                        new_sequence.append(sequence[index])
                    else:
                        new_sequence.append(sequence[index]*multiple)
                new_data.append(new_sequence)
            X_output.append(new_data)
        Y_output.extend(Y_data.tolist())

    print('X previous shape: {}, after shape: {}'.format(X_data.shape, np.array(X_output).shape))
    print('Y previous shape: {}, after shape: {}'.format(Y_data.shape, np.array(Y_output).shape))

    return np.array(X_output), np.array(Y_output)

def train_test_split(X_data, Y_data):
    set_size = 80  # (-400,-400) ... (400,400)
    total_set_count = int(len(X_data)/set_size)  # total_set_count = 35
    test_set_count = int(total_set_count*0.1)  # test_set_count = 3

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
    #model.add(Dropout(0.25))

    model.add(Conv2D(filters * 8, kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(filters * 16, kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(filters * 32, kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    #model.add(Dropout(0.5))
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

def train_save(model, X_train, X_test, Y_train, Y_test, train_time):
    MODEL_SAVE_FOLDER_PATH = './best_model/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)
    model_path = MODEL_SAVE_FOLDER_PATH + 'model_{}.hdf5'.format(train_time)

    best_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')

    # set_size = 80
    # total_set_count = int(len(X)/set_size)9.6087e-04
    # val_set_count = int(total_set_count*0.2)
    # epochs_size = 100
    #
    # X = X.reshape(total_set_count,set_size,375,5,1)
    # Y = Y.reshape(total_set_count,set_size,2)
    # print('X training set: {}'.format(X.shape))
    # print('Y training set: {}'.format(Y.shape))

    # loss = list()
    # val_loss = list()
    #for epochs in range(epochs_size):
        # X_val = list()
        # X_train = list()
        # Y_val = list()
        # Y_train = list()
        #
        # set_num = list(range(total_set_count))
        # random.shuffle(set_num)
        # for index, value in enumerate(set_num):
        #     if(index < val_set_count):
        #         X_val.append(X[value])
        #         Y_val.append(Y[value])
        #     else:
        #         X_train.append(X[value])
        #         Y_train.append(Y[value])
        #
        # X_val = np.array(X_val)
        # X_train = np.array(X_train)
        # Y_val = np.array(Y_val)
        # Y_train = np.array(Y_train)
        #
        # X_val = X_val.reshape(len(X_val) * set_size, 375, 5, 1)
        # X_train = X_train.reshape(len(X_train) * set_size, 375, 5, 1)
        # Y_val = Y_val.reshape(len(Y_val) * set_size, 2)
        # Y_train = Y_train.reshape(len(Y_train) * set_size, 2)
        #
        # print('X_train_shape: {}'.format(X_train.shape))
        # print('X_val_shape: {}'.format(X_val.shape))
        # print('Y_train_shape: {}'.format(Y_train.shape))
        # print('Y_val_shape: {}'.format(Y_val.shape))
        #
        # print('epoch: {}'.format(epochs))
        # history = model.fit(X_train, Y_train,
        #                     epochs=100,
        #                     batch_size=80,
        #                     validation_data=(X_val, Y_val),
        #                     verbose=2,
        #                     callbacks=[best_save])
        # loss.extend(history.history['loss'])
        # val_loss.extend(history.history['val_loss'])

    # fig, loss_ax = plt.subplots()
    # loss_ax.plot(loss, 'y', label='train loss')
    # loss_ax.plot(val_loss, 'r', label='val loss')
    # loss_ax.set_xlabel('epoch')
    # loss_ax.set_ylabel('loss')
    # loss_ax.legend(loc='upper left')
    # plt.show()

    history = model.fit(X_train, Y_train,
                        epochs=100,
                        batch_size=80,
                        validation_split=0.2,
                        verbose=2,
                        callbacks=[best_save])
    fig, loss_ax = plt.subplots()
    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')
    plt.show()

    test_loss = model.evaluate(X_test, Y_test)
    print('test_loss: ', test_loss)

def load_best_model(train_time):
    model = load_model('./best_model/model_{}.hdf5'.format(train_time), custom_objects={'my_loss_E1': my_loss_E1, })
    return model


# prepare data
submit = pd.read_csv('sample_submission.csv')
X_data = reshape(X_data, 'X')
Y_data = reshape(Y_data, 'Y')

# augment data and authenticate
fold = 3
original_data_count = len(X_data)
data_testID = random.randint(0,original_data_count-1)

X_data, Y_data = augmentation(X_data, Y_data, fold)
print('Original X: {}, Y: {}'.format(Y_data[data_testID][0], Y_data[data_testID][1]))
print('Modified X: {}, Y: {}'.format(Y_data[data_testID+2800*(fold-1)][0], Y_data[data_testID+2800*(fold-1)][1]))

for features in range(1,5):
    for sequence in range(375):
        if X_data[data_testID][sequence][features][0] != 0:
            print('Original sensor {} initial time: {}'.format(features, sequence))
            break
    for sequence in range(375):
        if X_data[data_testID+2800*(fold-1)][sequence][features][0] != 0:
            print('Modified sensor {} initial time: {}'.format(features, sequence))
            break

# plt.figure(figsize=(8,6))
# plt.plot(X_data[data_testID,:,1,0], label="Sensor #1_original")
# plt.plot(X_data[data_testID+2800*(fold-1),:,1,0], label="Sensor #1_modified")
# plt.plot(X_data[data_testID,:,2,0], label="Sensor #2_original")
# plt.plot(X_data[data_testID+2800*(fold-1),:,2,0], label="Sensor #2_modified")
# plt.plot(X_data[data_testID,:,3,0], label="Sensor #3_original")
# plt.plot(X_data[data_testID+2800*(fold-1),:,3,0], label="Sensor #3_modified")
# plt.plot(X_data[data_testID,:,4,0], label="Sensor #4_original")
# plt.plot(X_data[data_testID+2800*(fold-1),:,4,0], label="Sensor #4_modified")
#
# plt.xlabel("Time", labelpad=10, size=20)
# plt.ylabel("Acceleration", labelpad=10, size=20)
# plt.xticks(size=15)
# plt.yticks(size=15)
# plt.xlim(0, 400)
# plt.legend(loc=1)
# plt.show()

# split train/test data and check
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data)
pos_X = list()
pos_Y = list()
for position in Y_train:
    pos_X.append(position[0])
    pos_Y.append(position[1])
plt.figure()
plt.title('Train distribution')
plt.scatter(pos_X, pos_Y, alpha=.1)
plt.show()

pos_X = list()
pos_Y = list()
for position in Y_test:
    pos_X.append(position[0])
    pos_Y.append(position[1])
plt.figure()
plt.title('Test distribution')
plt.scatter(pos_X, pos_Y, alpha=.1)
plt.show()

# set model and train
model = set_model()
train_time = datetime.now().strftime("%m_%d_%H:%M")
train_save(model, X_train, X_test, Y_train, Y_test, train_time)  # train and save best model

# load best model
best_model = load_best_model(train_time)

# predict the unknown data
X_predict = reshape(X_predict, 'X')
Y_predict = best_model.predict(X_predict)

submit.iloc[:, 1] = Y_predict[:, 0]
submit.iloc[:, 2] = Y_predict[:, 1]
submit.to_csv('result/submit.csv', index = False)

