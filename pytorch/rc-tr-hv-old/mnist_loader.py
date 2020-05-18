
import glob
import torch
import numpy as np
import pandas as pd
import os
import re
import struct
from numpy import genfromtxt

from torch.utils.data import Dataset

class MNISTDataset( Dataset ): 

	def __init__( self, csv_path, mat, vec): 
		#read the dataset here.
		matfile = csv_path + mat
		vecfile = csv_path + vec

		self.data = []
		with open( matfile, 'r') as f:
			for l in f: 
				e = [ float(v) for v in l.split(',') ]
				self.data.append( e)

		self.labels =  [] 
		with open( vecfile, 'r') as f: 
			for l in f: 
				e = [ int (v) for v in l.split( ',' ) ]
				self.labels.append( e )
		print(' Done loading files here... ')

		#convert them to doubel np array here. 
		self.data = np.asarray( self.data, dtype=np.float64 )
		self.labels = np.asarray( self.labels, dtype=np.int)

		#print( 'Reshaping now... ')
		#print( self.data.shape )
		self.data = self.data.reshape( (self.data.shape[0], 1, 28, 28) )
		#print( self.data.shape )
		#print( self.labels.shape )
		
	def __len__( self ): 
		return len( self.labels )

	def __getitem__( self, index ): 
		return self.data[ index ], self.labels[ index ]

	def getDataset (self): 
		#print( 'unique labels: %s ' % (np.unique( self.labels )) )
		#print( 'no of samples: %d ' % (len( self.labels )) )
		#print( self.data.shape )
		return self.data, self.labels

def get_mnist_data_loader( cvs_path, batch_size, shuffle=False, 
		num_workers=4, pin_memory=False, 
		matFile='train_mat.txt', vecFile='train_vec.txt', sampler=None ):

        dataset = MNISTDataset( csv_path=cvs_path, mat=matFile, vec=vecFile)

        data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=pin_memory)

        data_loader_sampler = None
        if sampler is not None:
                data_loader_sampler = torch.utils.data.DataLoader(
                        dataset, batch_size=5000, shuffle=False, sampler=sampler,
                        num_workers=num_workers, pin_memory=pin_memory)


        return data_loader, data_loader_sampler
