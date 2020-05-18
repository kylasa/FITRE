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

from data_download import CIFAR10
#from data_download import CIFAR100

from torch.utils.data import Dataset

class TensorDataset( Dataset ): 
	def __init__( self, csv_path, fileFilter ): 
		self.transform = None
	
		#self.ones = torch.sparse.torch.eye( 10 ).type( torch.DoubleTensor)
		self.ones = torch.sparse.torch.eye( 10 ).type( torch.LongTensor )
		self.identity = np.identity( 10, dtype=np.float64 )
		t = {}
		#print (csv_path + "data_batch_*" )
		for i in glob.glob( csv_path + fileFilter ): 
			#print (i)
			a = self.unpickle( i ) 
			#print( len( a[ 'labels' ] ) )

			if len( t ) == 0: 
				t[ 'data' ] = a[ 'data' ]
				t[ 'labels' ] = np.array( a[ 'labels' ] )
			else:
				x = np.concatenate ((t[ 'data' ], a[ 'data' ]), axis=0)
				y = np.append( t[ 'labels' ], a[ 'labels' ] )

				t[ 'data' ] = x
				t[ 'labels' ] = y

			#break
			#print( len( t[ 'labels' ] ) )

		self.data = t[ 'data' ]
		self.labels = t[ 'labels' ]
					
		#print( t[ 'data' ].shape )
		#print( len(t[ 'labels' ]) )

	def unpickle( self, dfile ): 
		import cPickle
		with open( dfile, 'rb' ) as fo: 
			d = cPickle.load( fo )
		return d

	def __len__( self ): 
		return len( self.labels )

	def __getitem__ (self, index ): 
			x = self.data[ index ]
			y = self.labels[ index ]	

			#x = np.reshape( x, (32, 32, 3), order='C')	
			#x = np.reshape( x, (3, 32, 32), order='C').astype( np.float64)
			#return torch.from_numpy( x ), y
			
			x = np.reshape( x, (3, 32, 32), order='C').astype( np.float64) / 255.
			#x = np.reshape( x, (3, 32, 32), order='C').astype( np.float64)
			#return x, y.astype( np.float64) 
			#return x, self.ones.index_select(0, torch.LongTensor( [y] ) )
			#return x, self.identity[ y ]
			return x,  y

	def getSample( self, idx ): 
			x = self.data[ idx ]
			y = self.labels[ idx ]

			x = np.reshape( x, (len(idx), 3, 32, 32), order='C').astype( np.float64 ) / 255.
			#x = np.reshape( x, (len(idx), 3, 32, 32), order='C').astype( np.float64 )
			return x, self.identity[ y ]

class GaussianDataset( Dataset ): 
	def __init__( self, csv_path, fileFilter ): 
	
		self.identity = np.identity( 10, dtype=np.float64 )
		data = []
		labels = []
		for i in glob.glob( csv_path + fileFilter ): 
			a, b = self.readFile( i ) 

			if len( data ) == 0: 
				data = a
				labels = b
			else:
				#x = np.concatenate ((a, data), axis=0)
				#y = np.append( labels, b )
				data = np.concatenate( (data, a), axis=0 )
				labels = np.append (labels, b)

		self.data = data
		self.labels = labels

		print (len( self.data ) )
		print( len( self.labels) )
					
	def readFile( self, infile ): 
		my_data = genfromtxt( infile, delimiter=',' ) # pts * 3072 array
		s = re.split( '_|\.', infile)	
		print( s )
		return my_data, np.ones( my_data.shape[0], dtype=np.int ) * int(s[1]) 

	def __len__( self ): 
		return len( self.labels )

	def __getitem__(self, index): 
		x = self.data[ index ]
		x = np.reshape( x, (3, 32, 32), order='C').astype( np.float64)
		y = self.labels[ index ]
		#return x, self.identity[ y ]		
		return x,  y 

class MyDataset(Dataset): 

	def __init__( self ): 
		self.data = np.arange( 100 )
		
	def __getitem__( self, index ): 
		return self.data[ index ]

	def __len__(self): 
		return len( self.data )


def get_data_loader( batch_size=10,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False, sampler=None):

	dataset = MyDataset( )

	data_loader = torch.utils.data.DataLoader( 
		dataset, batch_size=batch_size, shuffle=shuffle, 
		num_workers=num_workers, pin_memory=pin_memory)

	data_loader_sampler = None
	if sampler is not None: 
		data_loader_sampler = torch.utils.data.DataLoader( 
			dataset, batch_size=10, shuffle=False, sampler=sampler,
			num_workers=num_workers, pin_memory=pin_memory)

	return data_loader, data_loader_sampler


def get_gaussian_data_loader( cvs_path, batch_size, shuffle=False, num_workers=4, 
									pin_memory=False, fileFilter='class_*', sampler = None, samplesize=100 ): 

	dataset = GaussianDataset( csv_path=cvs_path, fileFilter=fileFilter )

	data_loader = torch.utils.data.DataLoader( 
		dataset, batch_size=batch_size, shuffle=shuffle, 
		num_workers=num_workers, pin_memory=pin_memory)

	data_loader_sampler = None
	if sampler is not None: 
		data_loader_sampler = torch.utils.data.DataLoader( 
			dataset, batch_size=samplesize, shuffle=False, sampler=sampler,
			num_workers=num_workers, pin_memory=pin_memory)

	return data_loader, data_loader_sampler

def get_tensor_data_loader( cvs_path, batch_size, shuffle=False, num_workers=4, 
									pin_memory=False, fileFilter='data_batch_*', sampler=None ): 

	dataset = TensorDataset( csv_path=cvs_path, fileFilter=fileFilter )

	data_loader = torch.utils.data.DataLoader( 
		dataset, batch_size=batch_size, shuffle=shuffle, 
		num_workers=num_workers, pin_memory=pin_memory)

	data_loader_sampler = None
	if sampler is not None: 
		data_loader_sampler = torch.utils.data.DataLoader( 
			dataset, batch_size=batch_size * 5, shuffle=False, sampler=sampler,
			num_workers=num_workers, pin_memory=pin_memory)

	return data_loader, data_loader_sampler

def get_cifar_large_sampler(  dataset, batch_size, sampler, num_workers, pin_memory ): 
	return torch.utils.DataLoader( dataset, batch_size = batch_size, shuffle=True, sampler=sampler, 
				num_workers=num_workers, pin_memory=pin_memory )


def get_cifar_data_loader(csv_path,
                    batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False, 
							train=True, 
                    transform=None, sampler=None):

	dataset = CIFAR10( csv_path, train=train, transform=transform)

	data_loader = torch.utils.data.DataLoader( 
		dataset, batch_size=batch_size, shuffle=shuffle, 
		num_workers=num_workers, pin_memory=pin_memory)

	data_loader_sampler = None
	if sampler is not None: 
		data_loader_sampler = torch.utils.data.DataLoader( 
			dataset, batch_size=25000, shuffle=False, sampler=sampler,
			num_workers=num_workers, pin_memory=pin_memory)

	return data_loader, data_loader_sampler


def get_sampler( csv_path): 

	dataset = TensorDataset( csv_path=csv_path, fileFilter='data_batch_*')
	data_loader = torch.utils.data.DataLoader( 
		dataset, shuffle=True, num_workers=4 )

	return data_loader
	
