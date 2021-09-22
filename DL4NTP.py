import tensorflow as tf
from tensorflow import keras
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import datetime as dt
import time
import csv
import seaborn as sns
from datetime import datetime, date
import keras.backend as K
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

from random import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def readData(filename):
    print('Reading dataset...')
    with open(filename) as f:
        SANReN = f.readlines()
    return SANReN


def loadData(filenames, x):
    
    with open('SANREN.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read(x)+"\n")


def preprocess(data, inputs):
    '''
    Preprocesses the data.
    '''
    headings_line = data[0].split()
    # Merge 'Src', 'IP', and 'Addr:Port'
    headings_line[4:7] = [''.join(headings_line[4:7])]
    # Merge 'Dst', 'IP', and 'Addr:Port'
    headings_line[5:8] = [''.join(headings_line[5:8])]
    # Remove 'Flags', 'Tos', and 'Flows'.
    headings_line = headings_line[0:6] + headings_line[8:13]
    
    # Clean time-series data points.
    framedata = []
    print('Preprocessing data...')
    for i in range(1, inputs):
        data_line = data[i].split()
        #print(data_line)
        #print(i)

        if (data_line[0] == 'Summary:') or (data_line[0] == 'Time') or (data_line[0] == 'Total') or (data_line[0] == 'Date') or (data_line[0] == 'Sys:') or (len(data_line) < 15):
            pass

        else:
            if (data_line[10] == 'M'):
                pass
            elif ((data_line[11] == "M" or data_line[11] == 'G') and (data_line[13] == 'M' or data_line[13] == 'G') and (data_line[15] == 'M' or data_line[15] == 'G')):
                if (data_line[11] == 'G'):
                    data_line[10] = float(data_line[10])*100000000
                else:
                    data_line[10] = float(data_line[10])*1000000
                if (data_line[13] == 'G'):
                    data_line[12] = float(data_line[12])*100000000
                else:
                    data_line[12] = float(data_line[12])*1000000
                if (data_line[15] == 'G'):
                    data_line[14] = float(data_line[14])*100000000
                else:
                    data_line[14] = float(data_line[14])*1000000

                data_line = data_line[0:5] + data_line[6:7] + data_line[9:11] + \
                    data_line[12:13] + data_line[14:15] + data_line[16:17]
                framedata.append(data_line)
            # Bytes and BPS in megabytes\n"
            elif ((data_line[11] == "M" or data_line[11] == 'G') and (data_line[14] == 'M' or data_line[14] == 'G')):
                if (data_line[11] == 'G'):
                    data_line[10] = float(data_line[10])*100000000
                else:
                    data_line[10] = float(data_line[10])*1000000
                if (data_line[14] == 'G'):
                    data_line[13] = float(data_line[13])*100000000
                else:
                    data_line[13] = float(data_line[13])*1000000

                data_line = data_line[0:5] + data_line[6:7] + \
                    data_line[9:11] + data_line[12:14] + data_line[15:16]
                framedata.append(data_line)
            # Bytes and BPS in megabytes\n"
            elif ((data_line[12] == "M" or data_line[12] == 'G') and (data_line[12] == 'M' or data_line[12] == 'G')):
                if (data_line[12] == 'G'):
                    data_line[11] = float(data_line[11])*100000000
                else:
                    data_line[11] = float(data_line[11])*1000000
                if (data_line[14] == 'G'):
                    data_line[13] = float(data_line[13])*100000000
                else:
                    data_line[13] = float(data_line[13])*1000000

                data_line = data_line[0:5] + data_line[6:7] + \
                    data_line[9:12] + data_line[13:14] + data_line[15:16]
                framedata.append(data_line)
            elif (data_line[13] == 'M' or data_line[13] == 'G'):  # BPS measured in megabytes
                if (data_line[13] == 'G'):
                    data_line[12] = float(data_line[12])*100000000
                else:
                    data_line[12] = float(data_line[12])*1000000

                data_line = data_line[0:5] + data_line[6:7] + \
                    data_line[9:13] + data_line[14:15]
                framedata.append(data_line)

            elif data_line[11] == 'M':  # Bytes measured in megabytes
                data_line = data_line[0:5] + data_line[6:7] + \
                    data_line[9:11] + data_line[12:15]
                # Change M bytes into byte measurement.
                data_line[7] = float(data_line[7])*1000000
                framedata.append(data_line)

            else:  # No megabyte metrics
                data_line = data_line[0:5] + data_line[6:7] + data_line[9:14]
                framedata.append(data_line)

              # append each line to 'mother' array.
    # Convert Numpy array into Pandas dataframe.
    df = pd.DataFrame(np.array(framedata), columns=headings_line).copy()
    print('Data converted to Pandas dataframe.')
    return df


def format(df):
    '''
    Formats the dataframe column correctly. 
    '''
    print('Formatting dataframe columns...')
    df['Datetimetemp'] = df['Date'] + ' ' + \
        df['first-seen']  # Combine Date and first-seen
    df = df.astype({'Date': 'datetime64[ns]'})
    df = df.astype({'first-seen': 'datetime64[ns]'})

    df["Day"] = df['Date'].dt.dayofweek  # Created Day variable.
    df = df.astype({'Date': str})
    #df = df.astype({'first-seen': np.datetime64})
    df = df.astype({'Duration': np.float64})
    df = df.astype({"SrcIPAddr:Port": str})
    df = df.astype({"DstIPAddr:Port": str})
    df = df.astype({"Packets": np.float64})
    df = df.astype({"Bytes": np.float64})
    df = df.astype({"pps": np.float64})
    df = df.astype({"bps": np.float64})
    df = df.astype({"Bpp": np.float64})

    print('Defining new variables...')
    # Create binary Weekend variable.
    df['Weekend'] = 0
    df.loc[df['Day'] == 5, 'Weekend'] = 1
    df.loc[df['Day'] == 6, 'Weekend'] = 1

    # Insert combined Datetime at front of dataframe.
    df.insert(0, 'Datetime', df['Datetimetemp'])
    df['Datetime'] = df.Datetime.astype('datetime64[ns]')
    # Convert Datetime into an integer representation. This is a deprecated method.
    df['Datetime'] = df.Datetime.astype('int64')

    # Define university holiday calender
    holidays = pd.date_range(start='2020-1-1', end='2020-3-14', freq='1D')
    holidays = holidays.append(pd.date_range(
        start='2020-5-1', end='2020-5-9', freq='1D'))
    holidays = holidays.append(pd.date_range(
        start='2020-07-08', end='2020-08-02', freq='1D'))
    holidays = holidays.append(pd.date_range(
        start='2020-09-18', end='2020-09-27', freq='1D'))
    holidays = holidays.append(pd.date_range(
        start='2020-11-24', end='2020-12-31', freq='1D'))
    holidays = holidays.strftime("%Y-%m-%d").tolist()
    df['Holiday'] = 0

    for date in holidays:
        df.loc[df['Date'] == date, 'Holiday'] = 1

    # Delete unused columns.
    del df['Date']
    return df


def viewDistributions(df):
    '''
    Visualises the distributions of the explanatory variables. 
    '''
    # Explore individual categories
    groups = [0, 6, 7, 8, 9, 12, 13, 14]
    values = df.values
    i = 1
    # plot each column
    plt.figure()
    for group in groups:
        plt.subplot(len(groups), 1, i)

        plt.plot(values[:, group])
        plt.title(df.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()


def split(df):
    '''
    Splits the data into test and train, as well as the respective x and y sections of both subsets. 
    '''
    del df['first-seen']
    train, test = train_test_split(
        df, test_size=0.2, random_state = 13, shuffle = False)
    # Drop target variable from training data.
    x_train = train.drop(
        ['Datetimetemp', 'Bytes', 'SrcIPAddr:Port', 'DstIPAddr:Port', 'Proto'], axis=1).copy()
    # The double brackets are to keep Bytes in a pandas dataframe format, otherwise it will be pandas Series.
    y_train = train[['Bytes']].copy()
    x_test = test.drop(['Datetimetemp', 'Bytes', 'SrcIPAddr:Port',
                        'DstIPAddr:Port', 'Proto'], axis=1).copy()
    # The double brackets are to keep Bytes in a pandas dataframe format, otherwise it will be pandas Series.
    y_test = test[['Bytes']].copy()
    
    view = input("View the split of training and test data? [Y/N]\n")
    if (view == 'Y'):
        plt.figure(figsize=(40, 10))
        plt.title("Split of Test and Train Set using Bytes as Target Variable")
        plt.scatter(train['Datetime'], train['Bytes'], label='Training set')
        plt.scatter(test['Datetime'], test['Bytes'], label='Test set')
        plt.ylabel("Bytes")
        plt.xlabel("int64 Datetime")
        plt.legend()
        plt.show()

    return x_train, y_train, x_test, y_test


def scale(data):
    '''
    Scales data to the range 0 - 1, but this can be passed in as a parameter if we want. 
    '''
    # Scale training dating
    # scikit MinMixScaler allows all variables to be normalised between 0 and 1.
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Compute the minimum and maximum to be used for later scaling
    scaler.fit(data)
    # Scale features of X according to feature_range.
    scaled_data = scaler.transform(data)
    return scaled_data


def simpleLSTM(x_train, y_train, x_test, y_test, batch_size, epochs, neurons):
    '''
    Builds and trains the baseline LSTM model. 
    '''
    # We need to figure out how to reshape effectively. This is linked to the comment below. If the middle parameter here is 1 then batch_size is 1.
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, random_state = 42, shuffle = False)
    X_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    X_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    X_valid = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))
    Y_valid = y_valid.reshape((y_valid.shape[0], 1, y_valid.shape[1]))

    n_features = X_train.shape[2]
    model = Sequential()
    # The 1 parameter here is the number of timesteps - essentially the lag. It is linked to the comment above.
    model.add(LSTM(neurons, activation='sigmoid',
                   input_shape=(batch_size, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    print("Training Baseline LSTM...")
    tic = time.perf_counter()  # Time at start of training
    model.fit(X_train, y_train, epochs=epochs, verbose=1,
              validation_data=(X_valid, Y_valid))
    val_loss = model.history.history['val_loss']
    toc = time.perf_counter()  # Time at end of training
    loss_per_epoch = model.history.history['loss']
    train_yhat = model.predict(X_train, verbose=0)
    tic2 = time.perf_counter()
    test_yhat = model.predict(X_test, verbose=0)
    toc2 = time.perf_counter()

    val_yhat = model.predict(X_valid, verbose=0)

    simple_train_mae = mean_absolute_error(y_train, train_yhat)
    simple_test_mae = mean_absolute_error(y_test, test_yhat)
    simple_val_mae = mean_absolute_error(y_valid, val_yhat)
    simple_train_mse = mean_squared_error(y_train, train_yhat)
    simple_test_mse = mean_squared_error(y_test, test_yhat)
    simple_val_mse = mean_squared_error(y_valid, val_yhat)

    simple_lstm_train_time = toc - tic
    simple_lstm_prediction_time = toc2 - tic2

    simple_r2 = r2_score(y_test, test_yhat)
    simple_val_r2 = r2_score(y_valid, val_yhat)
    return loss_per_epoch, val_loss, train_yhat, test_yhat, val_yhat, simple_lstm_train_time, simple_lstm_prediction_time, simple_train_mae, simple_test_mae, simple_val_mae, simple_train_mse, simple_test_mse, simple_val_mse, simple_r2, simple_val_r2


def bidirectionalLSTM(x_train, y_train, x_test, y_test, batch_size, epochs, neurons):
    '''
    Builds and trains a bidirectional LSTM. 
    '''

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, random_state = 42, shuffle = False)
    X_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    X_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    X_valid = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))
    Y_valid = y_valid.reshape((y_valid.shape[0], 1, y_valid.shape[1]))
    n_features = X_train.shape[2]
    model = Sequential()
    model.add(Bidirectional(LSTM(neurons, return_sequences=False,
                                 activation="sigmoid"), input_shape=(batch_size, n_features)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    print("Training Bidirectional LSTM...")
    tic = time.perf_counter()
    model.fit(X_train, y_train, epochs=epochs, verbose=1,
              validation_data=(X_valid, Y_valid))
    val_loss = model.history.history['val_loss']

    toc = time.perf_counter()

    loss_per_epoch = model.history.history['loss']
    train_yhat = model.predict(X_train, verbose=0)
    tic2 = time.perf_counter()
    test_yhat = model.predict(X_test, verbose=0)
    toc2 = time.perf_counter()

    val_yhat = model.predict(X_valid, verbose=0)

    bidirectional_lstm_train_time = toc - tic
    bidirectional_lstm_prediction_time = toc2 - tic2

    bidirectional_train_mae = mean_absolute_error(y_train, train_yhat)
    bidirectional_test_mae = mean_absolute_error(y_test, test_yhat)
    bidirectional_val_mae = mean_absolute_error(y_valid, val_yhat)
    bidirectional_train_mse = mean_squared_error(y_train, train_yhat)
    bidirectional_test_mse = mean_squared_error(y_test, test_yhat)
    bidirectional_val_mse = mean_squared_error(y_valid, val_yhat)

    bidirectional_r2 = r2_score(y_test, test_yhat)
    bidirectional_val_r2 = r2_score(y_valid, val_yhat)

    return loss_per_epoch, val_loss, train_yhat, test_yhat, val_yhat, bidirectional_lstm_train_time, bidirectional_lstm_prediction_time, bidirectional_train_mae, bidirectional_test_mae, bidirectional_val_mae, bidirectional_train_mse, bidirectional_test_mse, bidirectional_val_mse, bidirectional_r2, bidirectional_val_r2


def stackedLSTM(x_train, y_train, x_test, y_test, batch_size, epochs, neurons):
    '''
    Builds and trains a stacked LSTM.
    '''
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, random_state = 42, shuffle = False)
    
    X_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    X_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    X_valid = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))
    Y_valid = y_valid.reshape((y_valid.shape[0], 1, y_valid.shape[1]))

    n_features = X_train.shape[2]
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, activation="sigmoid",
                   input_shape=(batch_size, n_features)))
    model.add(LSTM(neurons, return_sequences=True))
    model.add(LSTM(neurons))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    print("Training Stacked LSTM...")
    tic = time.perf_counter()
    model.fit(X_train, y_train, epochs=epochs, verbose=1,
              validation_data=(X_valid, Y_valid))
    val_loss = model.history.history['val_loss']

    toc = time.perf_counter()
    loss_per_epoch = model.history.history['loss']

    train_yhat = model.predict(X_train, verbose=0)

    tic2 = time.perf_counter()
    test_yhat = model.predict(X_test, verbose=0)
    toc2 = time.perf_counter()

    val_yhat = model.predict(X_valid, verbose=0)

    stacked_lstm_training_time = toc - tic
    stacked_lstm_prediction_time = toc2 - tic2

    stacked_train_mae = mean_absolute_error(y_train, train_yhat)
    stacked_test_mae = mean_absolute_error(y_test, test_yhat)
    stacked_val_mae = mean_absolute_error(y_valid, val_yhat)
    stacked_train_mse = mean_squared_error(y_train, train_yhat)
    stacked_test_mse = mean_squared_error(y_test, test_yhat)
    stacked_val_mse = mean_squared_error(y_valid, val_yhat)

    stacked_r2 = r2_score(y_test, test_yhat)
    stacked_val_r2 = r2_score(y_valid, val_yhat)

    return loss_per_epoch, val_loss, train_yhat, test_yhat, val_yhat, stacked_lstm_training_time, stacked_lstm_prediction_time, stacked_train_mae, stacked_test_mae, stacked_val_mae, stacked_train_mse, stacked_test_mse, stacked_val_mse, stacked_r2, stacked_val_r2


def view_yhat(y_train, yhat_train, y_test, yhat_test, name):
    '''
    Constructs a scatter plot of obs vs pred for training and test data.
    '''
    plt.scatter(y_train/1000000, y_unscale(y_train, yhat_train) /
                1000000, alpha=0.5, marker='.', label='Training set')
    plt.scatter(y_test/1000000, y_unscale(y_train, yhat_test) /
                1000000, alpha=0.5, marker='.', label='Test set')
    plt.plot([0, 600], [0, 600])
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.legend()
    plt.title(name)


def plotLoss(loss):
    '''
    Plots the loss graph of an LSTM based off model.history.history['loss']
    '''
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(len(loss)), loss)
    plt.show()


def y_unscale(y, yhat):
    '''
    Unscales a predicted y observation.
    '''
    Yscaler = MinMaxScaler(feature_range=(
        0, 1))  # apply same normalisation to response.
    Yscaler.fit(y)
    y_pred = Yscaler.inverse_transform(yhat)
    return y_pred


def plotDB():
    '''
    Creates a plot of Bytes vs Days.
    '''
    mon = df.loc[df['Day'] == 0, 'Bytes'].sum()
    tue = df.loc[df['Day'] == 1, 'Bytes'].sum()
    wed = df.loc[df['Day'] == 2, 'Bytes'].sum()
    thur = df.loc[df['Day'] == 3, 'Bytes'].sum()
    fri = df.loc[df['Day'] == 4, 'Bytes'].sum()
    sat = df.loc[df['Day'] == 5, 'Bytes'].sum()
    sun = df.loc[df['Day'] == 6, 'Bytes'].sum()

    byte_days = [mon, tue, wed, thur, fri, sat, sun]
    days = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
    plt.bar(days, byte_days)
                  
    plt.xlabel('Day of Week')
    plt.ylabel('Bytes')
    plt.xticks(np.arange(0.0, 6.0, step=1))  # Set label locations.
    # Set text labels.
    plt.xticks(np.arange(7), ['Mon', 'Tue', 'Wed',
               'Thurs', 'Fri', 'Sat', 'Sun'])
    plt.title('Bytes vs Day of Week')
    plt.show()


def plotHB():
    '''
    Plots holiday byte total vs non-holiday byte total.
    '''
    no = df.loc[df['Holiday'] == 0, 'Bytes'].sum()
    yes  = df.loc[df['Holiday'] == 1, 'Bytes'].sum()
    
    byte_days = [no/4, yes/3]
    days = ['No holiday', 'Holiday']
    plt.bar(days, byte_days)
                  
    plt.xlabel('Holiday')
    plt.ylabel('Bytes')
    plt.title('Bytes vs Holiday')
    plt.show()

def heatmap(data):
    '''
    Builds a sns correlation matrix.
    '''
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='CMRmap')
    plt.show()

def gridSearch(size, epoch_list, neuron_list, reps, x_train, y_train, x_test, y_test):
    '''
    Manual grid search across a range of epochs and neurons. Writes output to a file. 
    '''
    for epochs in epoch_list:
        for neurons in neuron_list:
                for j in range(reps):
                    
                    print("Now testing: ", epochs, neurons, j, size)
                    loss_simple, val_simple, yhat_train_simple, yhat_test_simple, yhat_val_simple, simple_lstm_train_time, simple_lstm_prediction_time,  simple_train_mae, simple_test_mae, simple_val_mae, simple_train_mse, simple_test_mse, simple_val_mse, simple_r2, simple_val_r2 = simpleLSTM(
                                    x_train, y_train, x_test, y_test, 1, epochs, neurons)

                    loss_bidirectional, val_bi, yhat_train_bi, yhat_test_bi, yhat_val_bi, bidirectional_lstm_train_time, bidirectional_lstm_prediction_time, bidirectional_train_mae, bidirectional_test_mae, bidirectional_val_mae, bidirectional_train_mse, bidirectional_test_mse, bidirectional_val_mse, bidirectional_r2, bidirectional_val_r2 = bidirectionalLSTM(
                                    x_train, y_train, x_test, y_test, 1, epochs, neurons)

                    loss_stacked, val_stacked, yhat_train_stacked, yhat_test_stacked, yhat_val_stacked, stacked_lstm_training_time, stacked_lstm_prediction_time, stacked_train_mae, stacked_test_mae,stacked_val_mae, stacked_train_mse, stacked_test_mse, stacked_val_mse, stacked_r2, stacked_val_r2 = stackedLSTM(
                                    x_train, y_train, x_test, y_test, 1, epochs, neurons)

                    data = [size, epochs, neurons, simple_lstm_train_time, bidirectional_lstm_train_time,
                                        stacked_lstm_training_time, simple_train_mae, simple_val_mae, bidirectional_train_mae, bidirectional_val_mae, stacked_train_mae, stacked_val_mae, simple_train_mse, simple_val_mse, bidirectional_train_mse, bidirectional_val_mse, stacked_train_mse, stacked_val_mse, simple_val_r2, bidirectional_val_r2, stacked_val_r2]

                    with open('train_data.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)

def optimalSimple(size, epochs, neurons, x_train, y_train, x_test, y_test):
    '''
    Runs the optimal simple LSTM model
    '''
    loss_simple, val_simple, yhat_train_simple, yhat_test_simple, yhat_val_simple, simple_lstm_train_time, simple_lstm_prediction_time,  simple_train_mae, simple_test_mae, simple_val_mae, simple_train_mse, simple_test_mse, simple_val_mse, simple_r2, simple_val_r2 = simpleLSTM(
                                    x_train, y_train, x_test, y_test, 1, epochs, neurons)
    
    data = ['Simple', size, epochs, neurons, simple_test_mae, simple_test_mse, simple_r2, simple_lstm_prediction_time]
    with open('test_data.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)

    plt.subplot(1, 2, 1)
    simple_df = pd.DataFrame({'y': list(np.array(scale(y_test))), 'y_pred': list(np.array(yhat_test_simple))})
    plt.plot(simple_df)
    plt.xlabel('Index')
    plt.ylabel('Bytes')
    plt.legend(['Observed', 'Predicted'])
    plt.subplot(1, 2, 2)
    simple_df2 = pd.DataFrame({'y_pred': list(np.array(yhat_test_simple)), 'y': list(np.array(scale(y_test)))})
    plt.plot(simple_df2)
    plt.xlabel('Index')
    plt.ylabel('Bytes')
    plt.legend(['Predicted', 'Observed'])
    plt.show()

def optimalBi(size, epochs, neurons, x_train, y_train, x_test, y_test):
    '''
    Runs the optimal bidirectional LSTM model
    '''
    loss_bidirectional, val_bi, yhat_train_bi, yhat_test_bi, yhat_val_bi, bidirectional_lstm_train_time, bidirectional_lstm_prediction_time, bidirectional_train_mae, bidirectional_test_mae, bidirectional_val_mae, bidirectional_train_mse, bidirectional_test_mse, bidirectional_val_mse, bidirectional_r2, bidirectional_val_r2 = bidirectionalLSTM(
                                    x_train, y_train, x_test, y_test, 1, epochs, neurons)
    
    data = ['Bidirectional', size, epochs, neurons, bidirectional_test_mae, bidirectional_test_mse, bidirectional_r2, bidirectional_lstm_prediction_time]
    with open('test_data.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)

    plt.subplot(1, 2, 1)
    simple_df = pd.DataFrame({'y': list(np.array(scale(y_test))), 'y_pred': list(np.array(yhat_test_bi))})
    plt.plot(simple_df)
    plt.xlabel('Index')
    plt.ylabel('Bytes')
    plt.legend(['Observed', 'Predicted'])
    plt.subplot(1, 2, 2)
    simple_df2 = pd.DataFrame({'y_pred': list(np.array(yhat_test_bi)), 'y': list(np.array(scale(y_test)))})
    plt.plot(simple_df2)
    plt.xlabel('Index')
    plt.ylabel('Bytes')
    plt.legend(['Predicted', 'Observed'])
    plt.show()

def optimalStacked(size, epochs, neurons, x_train, y_train, x_test, y_test):
    '''
    Runs the optimal stacked LSTM model.
    '''
    loss_stacked, val_stacked, yhat_train_stacked, yhat_test_stacked, yhat_val_stacked, stacked_lstm_training_time, stacked_lstm_prediction_time, stacked_train_mae, stacked_test_mae,stacked_val_mae, stacked_train_mse, stacked_test_mse, stacked_val_mse, stacked_r2, stacked_val_r2 = stackedLSTM(
                                    x_train, y_train, x_test, y_test, 1, epochs, neurons)
    data = ['Stacked', size, epochs, neurons, stacked_test_mae, stacked_test_mse, stacked_r2, stacked_lstm_prediction_time]
    with open('test_data.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)

    plt.subplot(1, 2, 1)
    simple_df = pd.DataFrame({'y': list(np.array(scale(y_test))), 'y_pred': list(np.array(yhat_test_stacked))})
    plt.plot(simple_df)
    plt.xlabel('Index')
    plt.ylabel('Bytes')
    plt.legend(['Observed', 'Predicted'])
    plt.subplot(1, 2, 2)
    simple_df2 = pd.DataFrame({'y_pred': list(np.array(yhat_test_stacked)), 'y': list(np.array(scale(y_test)))})
    plt.plot(simple_df2)
    plt.xlabel('Index')
    plt.ylabel('Bytes')
    plt.legend(['Predicted', 'Observed'])
    plt.show()

if __name__ == "__main__":

    #Input all of the SANREN files you wish to read from:
    data = ['SANREN040720.txt', 'SANREN050720.txt', 'SANREN060720.txt', 'SANREN070720.txt', 'SANREN080720.txt', 'SANREN090720.txt', 'SANREN100720.txt'] 
    
    #Create a sample with x number of bytes from each file above:
    loadData(data, 1000000)
    SANREN = readData('SANREN.txt')
    print(len(SANREN))
    size = 47758

    #Define the hyperparameter lists for the grid-search:
    epochs_list = [1, 2, 3, 4, 5, 6]
    neuron_list = [5, 100]

    df = preprocess(SANREN, size)
    df = format(df)

    view = input("Would you like to view the preliminary statistical analysis plots? [Y/N]\n")
    if (view == 'Y'):
      
        plotDB()
        plotHB()

        heatmap_df = df.drop(['Datetimetemp', 'SrcIPAddr:Port', 'DstIPAddr:Port', 'Proto', 'Day', 'Weekend', 'Holiday'], axis=1).copy()
        heatmap(heatmap_df)

        sns.kdeplot(df['Bytes'])
        plt.title("Density of Byte Values")
        plt.show()

        q0 = min(df['Bytes'])
        q1 = np.percentile(df['Bytes'], 25)
        q2 = np.percentile(df['Bytes'], 50)
        q3 = np.percentile(df['Bytes'], 75)
        q4 = max(df['Bytes'])

        print('Min: %.2f' % q0)
        print('Q1: %.2f' % q1)
        print('Median: %.2f' % q2)
        print('Q3: %.2f' % q3)
        print('Max: %.2f' % q4)

        viewDistributions(df)

    #Split and scale the data for LSTM input.
    x_train, y_train, x_test, y_test = split(df)
    x_train_scaled = scale(x_train)
    y_train_scaled = scale(y_train)
    x_test_scaled = scale(x_test)
    y_test_scaled = scale(y_test)

    #Create new file that LSTM training data can be written to:
    with open('train_data.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['dataset_size', 'epochs', 'neurons', 'simple_lstm_train_time','bidirectional_lstm_train_time', 'stacked_lstm_training_time', 'simple_train_mae', 'simple_val_mae', 'bidirectional_train_mae', 'bidirectional_val_mae', 'stacked_train_mae', 'stacked_val_mae', 'simple_train_mse', 'simple_val_mse', 'bidirectional_train_mse', 'bidirectional_val_mse', 'stacked_train_mse', 'stacked_val_mse', 'simple_val_r2', 'bi_val_r2', 'stacked_val_r2',])
    f.close()
    gridSearch(size, epochs_list, neuron_list, 1, x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled)

    #Based on results, define optimal model hyperparameters:
    x = input("Enter selected optimal simple LSTM epoch hyperparameter: ")
    y = input("Enter selected optimal simple LSTM neuron hyperparameter: ")
    print("Simple LSTM model succesfully defined.")
    hyperparams_simple = [int(x), int(y)]

    x = input("Enter selected optimal bidirectional LSTM epoch hyperparameter: ")
    y = input("Enter selected optimal bidirectional LSTM neuron hyperparameter: ")
    print("Bidirectional LSTM model succesfully defined.")
    hyperparams_bi = [int(x), int(y)]

    x = input("Enter selected optimal stacked LSTM epoch hyperparameter: ")
    y = input("Enter selected optimal stacked LSTM neuron hyperparameter: ")
    print("Stacked LSTM model succesfully defined.")
    hyperparams_stack = [int(x), int(y)]

    #Create new file that LSTM test data can be written to:
    with open('test_data.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['name', 'size', 'epochs', 'neurons', 'test_mae', 'test_mse', 'r2', 'lstm_prediction_time'])
    f.close()

     
    optimalSimple(size, hyperparams_simple[0], hyperparams_simple[1], x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled)
    optimalBi(size, hyperparams_bi[0], hyperparams_bi[1], x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled)
    optimalStacked(size, hyperparams_stack[0], hyperparams_stack[1], x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled)