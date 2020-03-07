from keras.preprocessing import TimeseriesGenerator
from math import ceil

def getTrainTestGenerator(data, length=50, batch_size=20, Xcol, ycol, proportionTrainTest=0.8):
    '''Generates a generator for a train and test set, according to the parameters set (length, batch_size and proportion.
    Xcol indicates the columns to be used as input, ycol indicates the columns to be used as output.
    Returns a generator for training, and a generator for testing'''
    split_index = ceil(proportionTrainTest * len(data))
    genTrain = TimeseriesGenerator( data=data[Xcol], 
                                    targets=data[ycol],     
                                    length=length, 
                                    batch_size=batch_size, 
                                    end_index=split_index - 1)
    genTest = TimeseriesGenerator(  data=data[Xcol], 
                                    targets=data[ycol], 
                                    length=length, 
                                    start_index=split_index - length)
    return genTrain, genTest

def scale

def timeseriesSplitTestTrainDF(df, proportion=0.8):
	'''Splits time-series numpy df in a training and test set, according to the proportion set (default = 0.8)
	Returns train and test dfs'''
	split_index = ceil(proportion * len(df))
	train = df.iloc[:indexSplit]
	test = df.iloc[indexSplit:]
	return train, test
    

genTrain = TimeseriesGenerator(data = np.df(data), targets=np.df(data)[:,0], length=length, batch_size=32, end_index=split_index - 1)
genTest = TimeseriesGenerator(data = np.df(data), targets=np.df(data)[:,0], length=length, batch_size=32, start_index=split_index - length )