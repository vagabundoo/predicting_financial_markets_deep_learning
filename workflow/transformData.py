import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from math import ceil
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import splitSequence as sq


def getTrainTestGenerator(data, Xcol, ycol, length=50, batch_size=20,proportionTrainTest=0.8, stride=1):
    '''Generates a generator for a train and test set, according to the parameters set (length, batch_size and proportion.
    Xcol indicates the columns to be used as input, ycol indicates the columns to be used as output.
    Returns a generator for training, and a generator for testing'''
    split_index = ceil(proportionTrainTest * len(data))
    genTrain = TimeseriesGenerator(data=np.array(data[Xcol]),
                                   targets=np.array(data[ycol]),
                                   length=length,
                                   batch_size=batch_size,
                                   stride=stride,
                                   end_index=split_index - 1)
    genTest = TimeseriesGenerator(data=np.array(data[Xcol]),
                                  targets=np.array(data[ycol]),
                                  stride=stride,
                                  length=length,
                                  start_index=split_index - length)
    return genTrain, genTest


def timeseriesSplitTestTrainDF(df, proportion=0.8):
    '''Splits time-series numpy df in a training and test set, according to the proportion set (default = 0.8)
    Returns train and test dfs'''
    split_index = ceil(proportion * len(df))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test


def minMaxScaleOnTrain(df, proportion=0.8):
    '''Gets minimum and maximum from training proportion of data (as determined by proportion-parameter)
    Returns full df (train and test) scaled to training data min-max'''
    train, test = timeseriesSplitTestTrainDF(df, proportion=0.8)
    scaler = MinMaxScaler()
    scaler.fit(train)
    transformedData = scaler.transform(df)
    df = pd.DataFrame(data=transformedData, columns=df.columns, index=df.index)     
    return df

def StandardizeOnTrain(df, proportion=0.8):
    '''Standardize on training proportion of data (as determined by proportion-parameter)
    Returns full df (train and test) scaled to training data mean and standard deviation'''
    train, test = timeseriesSplitTestTrainDF(df, proportion=0.8)
    scaler = StandardScaler()
    scaler.fit(train)
    transformedData = scaler.transform(df)
    df = pd.DataFrame(data=transformedData, columns=df.columns, index=df.index)     
    return df
