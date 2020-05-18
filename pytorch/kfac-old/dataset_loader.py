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
import os
import re
import struct
from numpy import genfromtxt

from torch.utils.data import Dataset

class TensorDataset( Dataset ): 
	def __init__( self, csv_path, fileFilter ): 

		t = {}
		for i in glob.glob( csv_path + fileFilter ): 
			print( 'Reading %s ' % (i) )
			a = self.unpickle( i ) 

			if len( t ) == 0: 
				t[ 'data' ] = a[ 'data' ]
				t[ 'labels' ] = np.array( a[ 'labels' ] )
			else:
				x = np.concatenate ((a[ 'data' ], t[ 'data' ]), axis=0)
				y = np.append( t[ 'labels' ], a[ 'labels' ] )

				t[ 'data' ] = x
				t[ 'labels' ] = y

		self.data = t[ 'data' ]
		self.labels = t[ 'labels' ]

		self.data = self.data.reshape((self.data.shape[0], 3, 32, 32))
		#self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
					

	def unpickle( self, dfile ): 
		import cPickle
		with open( dfile, 'rb' ) as fo: 
			d = cPickle.load( fo )
		return d

	def getDataset (self): 
		print( 'unique labels: %s ' % (np.unique( self.labels )) )
		print( 'no of samples: %d ' % (len( self.labels )) )
		return self.data, self.labels

