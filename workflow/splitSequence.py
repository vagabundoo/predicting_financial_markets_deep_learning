import numpy as np

def splitTimeSequence(sequence, n_steps):
	'''Splits time series into X and y, with n_steps determining the length of each X. 
	Returns numpy arrays of X and y'''
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def splitSeqGetX(sequence, n_steps):
	'''Splits time series into X and y, with n_steps determining length. Returns only the X. '''
	return splitTimeSequence(sequence, n_steps)[0]

def splitSeqGety(sequence, n_steps):
	'''Splits time series into X and y, with n_steps determining length. Returns only the y. '''
	return splitTimeSequence(sequence, n_steps)[1]

def timeseriesSplitTestTrain(array, proportion=0.8):
	'''Splits time-series numpy array in a training and test set, according to the proportion set (default = 0.8)
	Returns train and test arrays'''
	indexSplit = round(len(array) * 0.8)
	train = array[:indexSplit]
	test = array[indexSplit:]
	return train, test

def shapeXinputforColumns(array, n_steps, col_indexes):
	X = []
	#col_index = [1,2,3,4]
	for i in col_indexes:
		X.append(splitSeqGetX(array[:,i], n_steps))
	return np.dstack(X)

