import tensorflow as tf
from tensorflow import keras
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import datetime as dt

from datetime import datetime, date
import keras.backend as K
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers  import LSTM
from tensorflow.keras.layers  import TimeDistributed
from tensorflow.keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from numpy import array
from numpy import cumsum
from pandas.tseries.holiday import USFederalHolidayCalendar
from random import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

class DL4NTP:

    def readData(filename):
        print('Reading dataset...')
        with open(filename) as f:
            SANReN = f.readlines()
        return SANReN

    def preprocess(data):
        headings_line = data[0].split()
        headings_line[4:7] = [''.join(headings_line[4:7])] #Merge 'Src', 'IP', and 'Addr:Port' 
        headings_line[5:8] = [''.join(headings_line[5:8])] #Merge 'Dst', 'IP', and 'Addr:Port' 
        headings_line = headings_line[0:6] + headings_line[8:13] #Remove 'Flags', 'Tos', and 'Flows'.

            #Clean time-series data points. 
        framedata = []
        print('Preprocessing data...')
        for i in range(1, 1001):
            data_line = data[i].split()
        
            if ((data_line[11] == "M" or data_line[11] == 'G') and (data_line[13] == 'M' or data_line[13] == 'G') and (data_line[15] == 'M' or data_line[15] == 'G')):
                if (data_line[11] == 'G'): data_line[10] = float(data_line[10])*100000000
                else: data_line[10] = float(data_line[10])*1000000
                if (data_line[13] == 'G'): data_line[12] = float(data_line[12])*100000000
                else: data_line[12] = float(data_line[12])*1000000
                if (data_line[15] == 'G'): data_line[14] = float(data_line[14])*100000000
                else: data_line[14] = float(data_line[14])*1000000

                data_line = data_line[0:5] + data_line[6:7] + data_line[9:11] + data_line[12:13] + data_line[14:15] + data_line[16:17]

            elif ((data_line[11] == "M" or data_line[11] == 'G') and (data_line[14] == 'M' or data_line[14] == 'G')): #Bytes and BPS in megabytes\n"
                if (data_line[11] == 'G'): data_line[10] = float(data_line[10])*100000000
                else: data_line[10] = float(data_line[10])*1000000
                if (data_line[14] == 'G'): data_line[13] = float(data_line[13])*100000000
                else: data_line[13] = float(data_line[13])*1000000
                
                data_line = data_line[0:5] + data_line[6:7] + data_line[9:11] + data_line[12:14] + data_line[15:16]
            
            elif ((data_line[12] == "M" or data_line[12] == 'G') and (data_line[12] == 'M' or data_line[12] == 'G')): #Bytes and BPS in megabytes\n"
                if (data_line[12] == 'G'):data_line[11] = float(data_line[11])*100000000
                else: data_line[11] = float(data_line[11])*1000000
                if (data_line[14] == 'G'): data_line[13] = float(data_line[13])*100000000
                else: data_line[13] = float(data_line[13])*1000000
                
                data_line = data_line[0:5] + data_line[6:7] + data_line[9:12] + data_line[13:14] + data_line[15:16]
                
            elif (data_line[13] == 'M' or data_line[13] == 'G'): #BPS measured in megabytes
                if (data_line[13] == 'G'): data_line[12] = float(data_line[12])*100000000
                else: data_line[12] = float(data_line[12])*1000000
                
                data_line = data_line[0:5] + data_line[6:7] + data_line[9:13] + data_line[14:15]
            
            elif data_line[11] == 'M': #Bytes measured in megabytes
                data_line = data_line[0:5] + data_line[6:7] + data_line[9:11] + data_line[12:15]
                data_line[7] = float(data_line[7])*1000000 #Change M bytes into byte measurement. 
                
            else: #No megabyte metrics
                data_line = data_line[0:5] + data_line[6:7] + data_line[9:14]
            
            framedata.append(data_line) #append each line to 'mother' array.
        #Convert Numpy array into Pandas dataframe.
        df = pd.DataFrame(np.array(framedata), columns=headings_line) 
        print('Data converted to Pandas dataframe.')
        return df

    def format(df):
        print('Formatting dataframe columns...')
        df['Datetimetemp'] = df['Date'] + ' ' + df['first-seen'] #Combine Date and first-seen
        df = df.astype({'Date': 'datetime64[ns]'})
        df["Day"] = df['Date'].dt.dayofweek #Created Day variable.
        df = df.astype({'first-seen': np.datetime64})
        df = df.astype({'Duration': np.float64})
        df = df.astype({"SrcIPAddr:Port": str})
        df = df.astype({"DstIPAddr:Port": str})
        df = df.astype({"Packets": np.int64})
        df = df.astype({"Bytes": np.float64})
        df = df.astype({"pps": np.float64})
        df = df.astype({"bps": np.float64})
        df = df.astype({"Bpp": np.float64})

        print('Defining new variables...')
        #Create binary Weekend variable.
        df['Weekend'] = 0
        df.loc[df['Day'] == 5 , 'Weekend'] = 1
        df.loc[df['Day'] == 6 , 'Weekend'] = 1

        #Insert combined Datetime at front of dataframe.
        df.insert(0, 'Datetime', df['Datetimetemp'])
        df['Datetime'] = df.Datetime.astype('datetime64[ns]')
        df['Datetime'] = df.Datetime.astype('int64') #Convert Datetime into an integer representation. This is a deprecated method. 

        #Define university holiday calender
        holidays = pd.date_range(start='2020-1-1', end='2020-3-14', freq = '1D')
        holidays = holidays.append(pd.date_range(start='2020-5-1', end='2020-5-9', freq='1D'))
        holidays = holidays.append(pd.date_range(start='2020-07-08', end='2020-08-02', freq='1D'))
        holidays = holidays.append(pd.date_range(start='2020-09-18', end='2020-09-27', freq='1D'))
        holidays = holidays.append(pd.date_range(start='2020-11-24', end='2020-12-31', freq='1D'))
        #sprint(df['Date'])
        #Add Holiday column to dataframe.
        df['Holiday'] = 0
        df.loc[(df['Date']) == any(holidays.date), 'Holiday'] = 1 #Can't get this to work

        #Delete unused columns.
        del df['Date']
        del df['first-seen']
        return df

    def viewDistributions(df):
        #Explore individual categories
        groups = [1, 5, 6, 7, 8, 9]
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
        train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        x_train = train.drop(['Datetimetemp', 'Bytes', 'SrcIPAddr:Port', 'DstIPAddr:Port', 'Proto'],axis=1).copy() #Drop target variable from training data. 
        y_train = train[['Bytes']].copy() # The double brackets are to keep Bytes in a pandas dataframe format, otherwise it will be pandas Series.
        x_test = test.drop(['Datetimetemp', 'Bytes', 'SrcIPAddr:Port', 'DstIPAddr:Port', 'Proto'],axis=1).copy()
        y_test = test[['Bytes']].copy() # The double brackets are to keep Bytes in a pandas dataframe format, otherwise it will be pandas Series.
        #print('X train shape', x_train.shape)
        #print(y_train.shape)
        view = input("View the split of training and test data? [Y/N]\n")
        if (view == 'Y'):
            plt.figure(figsize=(40,10))
            plt.title("Split of Test and Train Set using Bytes as Target Variable")
            plt.scatter(train['Datetime'],train['Bytes'],label='Training set')
            plt.scatter(test['Datetime'],test['Bytes'],label='Test set')
            plt.legend()
            plt.show()

        return x_train, y_train, x_test, y_test

    def scale(data):
        #Scale training dating
        scaler = MinMaxScaler(feature_range=(0, 1)) # scikit MinMixScaler allows all variables to be normalised between 0 and 1.
        scaler.fit(data) #Compute the minimum and maximum to be used for later scaling
        scaled_data = scaler.transform(data) #Scale features of X according to feature_range.
        return scaled_data

    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def simpleLSTM(x_train, y_train, x_test, y_test, batch_size, epochs, neurons):

        X_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1])) #We need to figure out how to reshape effectively. This is linked to the comment below. If the middle parameter here is 1 then batch_size is 1.
        X_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

        n_features = X_train.shape[2]
        model = Sequential()
        model.add(LSTM(neurons, activation='relu', input_shape=(batch_size, n_features))) #The 1 parameter here is the number of timesteps - essentially the lag. It is linked to the comment above. 
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs = epochs, verbose = 0)

        loss_per_epoch = model.history.history['loss']
        train_yhat = model.predict(X_train, verbose = 0)
        test_yhat = model.predict(X_test, verbose = 0)

        return loss_per_epoch, train_yhat, test_yhat

    def bidirectionalLSTM(x_train, y_train, x_test, y_test, batch_size, epochs, neurons):
        X_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1])) 
        X_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

        n_features = X_train.shape[2]
        model = Sequential() 
        model.add(Bidirectional(LSTM(neurons, return_sequences=False, activation="sigmoid"), input_shape=(batch_size, n_features)))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam') 

        model.fit(X_train, y_train, epochs = epochs, verbose = 0)

        loss_per_epoch = model.history.history['loss']
        train_yhat = model.predict(X_train, verbose = 0)
        test_yhat = model.predict(X_test, verbose = 0)

        return loss_per_epoch, train_yhat, test_yhat

    def view_yhat(y_train_scaled, yhat_train, y_test_scaled, yhat_test):
        plt.scatter(y_train_scaled, yhat_train,  label='Training set')
        plt.scatter(y_test_scaled, yhat_test, label='Test set')
        #plt.plot([min(y_train_scaled), max(y_train_scaled)], [min(yhat_train), max(yhat_train)],color='green',linewidth=2) #This was plotting a straight line through the graph.
        plt.legend()
        plt.xlabel('Observed Value')
        plt.ylabel('Predicted Value')
        plt.title('Training and Test Set: Predicted Values vs Observed Values ') 
        
        #plt.show()

    if __name__ == "__main__":
        SANREN = readData('SANREN_large.txt')
        df = preprocess(SANREN)
        df = format(df)

        view = input("View the distribution of the explanatory features? [Y/N]\n")
        if (view == 'Y'):
            viewDistributions(df)

        x_train, y_train, x_test, y_test = split(df)

        x_train_scaled = scale(x_train)
        y_train_scaled = scale(y_train)
        x_test_scaled = scale(x_test)
        y_test_scaled = scale(y_test)

        loss_simple, yhat_train_simple, yhat_test_simple = simpleLSTM(x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, 1, 100, 50)
        loss_bidirectional, yhat_train_bi, yhat_test_bi = bidirectionalLSTM(x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, 1, 100, 50)
        
        view = input("View the predicted yhat values for the test and training sets? [Y/N]\n")
        if (view == 'Y'):
            view_yhat(y_train_scaled, yhat_train_simple, y_test_scaled, yhat_test_simple)
            view_yhat(y_train_scaled, yhat_train_bi, y_test_scaled, yhat_test_bi)
            plt.show()

            view = input("View the loss graph of the simple LSTM\'s training process?")

            if view == 'Y':
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.plot(range(len(loss_simple)), loss_simple)
                plt.show()


        
       
        
    

        
  
        
       
        

        
        
        