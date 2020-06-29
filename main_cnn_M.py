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

X_data = np.loadtxt('new_train_features_MV.csv',skiprows=1,delimiter=',')
Y_data = np.loadtxt('new_train_target_MV.csv',skiprows=1,delimiter=',')
X_predict = np.loadtxt('test_features.csv',skiprows=1,delimiter=',')

def reshape(data, X_or_Y):
    if(X_or_Y == 'X'):
        data = data[:, 1:]
        data = data.reshape(int(len(data)/375),375,5,1)
        print('reshaped X: {}'.format(data.shape))
    elif(X_or_Y == 'Y'):
        data = data[:, 3]
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
    set_size = 35  # (25.0, 0.2) ... (175.0, 1.0)
    total_set_count = int(len(X_data)/set_size)
    test_set_count = int(total_set_count*0.1)

    X_test = X_data[:test_set_count * set_size]
    X_train = X_data[test_set_count * set_size:]
    print('X_train_shape: {}'.format(X_train.shape))
    print('X_test_shape: {}'.format(X_test.shape))

    Y_test = Y_data[:test_set_count * set_size]
    Y_train = Y_data[test_set_count * set_size:]
    print('Y_train_shape: {}'.format(Y_train.shape))
    print('Y_test_shape: {}'.format(Y_test.shape))

    return X_train, X_test, Y_train, Y_test

def my_loss_E2(y_true, y_pred):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])
    return K.mean(K.square(divResult))

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
    model.add(Dropout(0.25))

    model.add(Conv2D(filters * 8, kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(filters * 16, kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(filters * 32, kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1))

    optimizer = keras.optimizers.Adam()
    model.compile(loss=my_loss_E2,
                  optimizer=optimizer,
                  )
    model.summary()

    return model

def train_save(model, X_train, Y_train, train_time):
    MODEL_SAVE_FOLDER_PATH = './best_model/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)
    model_path = MODEL_SAVE_FOLDER_PATH + 'M_{}.hdf5'.format(train_time)

    best_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')

    history = model.fit(X_train, Y_train,
                        epochs=100,
                        batch_size=70,
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



def load_best_model(train_time):
    model = load_model('./best_model/M_{}.hdf5'.format(train_time), custom_objects={'my_loss_E2': my_loss_E2, })
    return model


# prepare data
submit = pd.read_csv('sample_submission.csv')
X_data = reshape(X_data, 'X')
Y_data = reshape(Y_data, 'Y')

# augment data and authenticate
fold = 1
original_data_count = len(X_data)
data_testID = random.randint(0,original_data_count-1)

X_data, Y_data = augmentation(X_data, Y_data, fold)
print('Original M: {}'.format(Y_data[data_testID]))
print('Modified M: {}'.format(Y_data[data_testID+2800*(fold-1)]))

for features in range(1,5):
    for sequence in range(375):
        if X_data[data_testID][sequence][features][0] != 0:
            print('Original sensor {} initial time: {}'.format(features, sequence))
            break
    for sequence in range(375):
        if X_data[data_testID+2800*(fold-1)][sequence][features][0] != 0:
            print('Modified sensor {} initial time: {}'.format(features, sequence))
            break

# split train/test data and check
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data)

# set model and train
model = set_model()
train_time = datetime.now().strftime("%m_%d_%H:%M")
train_save(model, X_train, Y_train, train_time)  # train and save best model

# load best model
best_model = load_best_model(train_time)

# evaluate test data loss
test_loss = model.evaluate(X_test, Y_test)
print('test_loss: ', test_loss)

# predict the unknown data and make submit file
X_predict = reshape(X_predict, 'X')
Y_predict = best_model.predict(X_predict)
submit.iloc[:, 3] = Y_predict[:, 0]
submit.to_csv('result/submit_M_{}_{}.csv'.format(test_loss, train_time), index = False)

# save renamed best model
best_model.save('./best_model/M_{}_{}.hdf5'.format(test_loss, train_time))
os.remove('./best_model/M_{}.hdf5'.format(train_time))

