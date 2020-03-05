# univariate data preparation
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
	return split_sequence(sequence, n_steps)[0]

def splitSeqGety(sequence, n_steps):
	'''Splits time series into X and y, with n_steps determining length. Returns only the y. '''
	return split_sequence(sequence, n_steps)[1]

def timeseriesSplitTestTrain(nparray, proportion=0.8):
	indexSplit = round(len(array) * 0.8)
	train = array[:indexSplit]
	test = array[indexSplit:]
	return train, test