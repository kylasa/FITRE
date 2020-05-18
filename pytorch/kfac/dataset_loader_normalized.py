"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.

[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
[2]: https://gist.githubusercontent.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb/raw/648f322aafe646f9e8f2c9636275701a90da36a6/data_loader.py
[3]: https://www.kaggle.com/mratsim/starting-kit-for-pytorch-deep-learning
"""

import glob
import torch
import numpy as np
import pandas as pd
import re
from numpy import genfromtxt

#from data_download import CIFAR10
#from data_download import CIFAR100

from torch.utils.data import Dataset

class CIFARReader: 

	@staticmethod
	def getNormalizedDataset( train_filter, test_filter ): 
		t = {}
		print( train_filter )
		for i in glob.glob( train_filter ): 
			print( i )
			with open( i , 'rb' ) as fo: 
				import cPickle
				a = cPickle.load( fo )

				if len( t ) == 0: 
					t[ 'traindata' ] = a[ 'data' ]
					t[ 'trainlabels' ] = np.array( a[ 'labels' ] )
				else:
					x = np.concatenate ((a[ 'data' ], t[ 'traindata' ]), axis=0)
					y = np.append( a[ 'labels' ], t[ 'trainlabels' ] )

					t[ 'traindata' ] = x
					t[ 'trainlabels' ] = y

		trainSize = t[ 'traindata' ].shape[0]

		for i in glob.glob( test_filter ): 
			with open( i , 'rb' ) as fo: 
				import cPickle
				a = cPickle.load( fo )

				t[ 'testdata' ] = a[ 'data' ]
				t[ 'testlabels' ] = a[ 'labels' ]

		consolidated = np.concatenate( (t[ 'traindata' ], t['testdata']), axis=0 ).astype( np.float64 )
		consolidated_sq = np.square( consolidated )
		column_norms = np.sum( consolidated_sq, axis=0 )
		column_norms = np.reciprocal( np.sqrt( column_norms ) )
		column_norms = np.tile( column_norms, (consolidated.shape[0], 1) )

		normalized_data = np.multiply( consolidated, column_norms )

		return normalized_data[ 0:trainSize, : ], t[ 'trainlabels' ], normalized_data[ trainSize:, : ], t[ 'testlabels' ]
		


class TensorDatasetNormalized( Dataset ): 

	def __init__( self, features, labels): 
		self.transform = None
	
		self.ones = torch.sparse.torch.eye( 10 ).type( torch.LongTensor )
		self.identity = np.identity( 10, dtype=np.float64 )

		self.data = features
		self.labels = labels
					
	def __len__( self ): 
		return len( self.labels )

	def __getitem__ (self, index ): 
			x = self.data[ index ]
			y = self.labels[ index ]	
			x = np.reshape( x, (3, 32, 32), order='C').astype( np.float64)
			return x,  y

	def getSample( self, idx ): 
			x = self.data[ idx ]
			y = self.labels[ idx ]
			x = np.reshape( x, (len(idx), 3, 32, 32), order='C').astype( np.float64 )
			return x, self.identity[ y ]

def get_cifar_large_sampler_normalized(  dataset, batch_size, sampler, num_workers=4, pin_memory=False ):  
   return torch.utils.DataLoader( dataset.dataset, batch_size = batch_size, shuffle=True, sampler=sampler, 
            num_workers=num_workers, pin_memory=pin_memory )

def getNormalizedCIFAR( cvs_path, batch_size, shuffle=False, num_workers=4, 
								pin_memory=False, fileFilter='data_batch_*', sampler=None, train_transform=None, test_transform=None ): 

	train, trainlabels, test, testlabels = CIFARReader.getNormalizedDataset( cvs_path + 'data_batch_*', cvs_path + 'test_batch' )

	'''
	print( 'testing.... ')
	traincol = train[ :, 0 ]
	print( traincol.shape )
	testcol = test[ :, 0 ]
	print( testcol.shape )

	c = np.concatenate( (traincol, testcol), axis=0 )
	s = np.square( c )
	total = np.sum( s )
	print( np.sqrt( total ) )
	'''	

	trainData = TensorDatasetNormalized( train, trainlabels )
	testData = TensorDatasetNormalized( test, testlabels)

	train_data_loader = torch.utils.data.DataLoader( 
		trainData, batch_size=batch_size, shuffle=shuffle, 
		num_workers=num_workers, pin_memory=pin_memory)

	data_loader_sampler = None
	if sampler is not None: 
		data_loader_sampler = torch.utils.data.DataLoader( 
			trainData, batch_size=batch_size, shuffle=False, sampler=sampler,
			num_workers=num_workers, pin_memory=pin_memory)

	test_data_loader = torch.utils.data.DataLoader( 
		testData, batch_size=batch_size, shuffle=shuffle, 
		num_workers=num_workers, pin_memory=pin_memory)

	return train_data_loader, data_loader_sampler, test_data_loader

