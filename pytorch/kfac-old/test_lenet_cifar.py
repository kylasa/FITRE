__author__ = 'Sudhir Kylasa'

import os
import datetime
import sys
import numpy as np
import math

import torch
from torch import nn
from torch.autograd import Variable

from dataset_loader import TensorDataset
from lenet_cifar_kfac import LeNetCIFARKFAC
from kfac_utils import KFACUtils

#torch.cuda.set_device( 1 )

#Read the dataset here. 
data_path = '/scratch/skylasa/solvers/cnn-data/cifar-10-batches-py/'
train = TensorDataset( csv_path=data_path, fileFilter='data_batch_1' )
train_data, train_labels = train.getDataset ()

train_data = torch.from_numpy( train_data ).type(torch.cuda.DoubleTensor)
train_labels = torch.from_numpy( train_labels ).type(torch.cuda.LongTensor)

BATCH_SIZE = 1000
bias = True

kfac  = KFACUtils( ksize = 5, layers = ['conv', 'conv', 'lin', 'lin', 'lin' ], batchSize = BATCH_SIZE, bias=bias, debug=True, gamma = 0.1)

model = LeNetCIFARKFAC( num_classes = 10, activation='softplus', batchSize = BATCH_SIZE, kfac=kfac, bias = bias)
model.double ()
model.cuda ()
model.initZeroWeights ()
model.initWeights( 0.1 ) 

model.initKFACStorage ()
model.startRecording ()

criterion = nn.CrossEntropyLoss (size_average=False)
model.setLossFunction( criterion )

'''
train_data[ 0, :, :, : ] = 1; 

print( train_data[ 0, :, :, : ].shape )
print( torch.unsqueeze( train_data[ 0, :, :, : ], 0 ).shape )
print( torch.from_numpy( np.asarray( [ train_labels[ 0 ] ] ) ) )
print( torch.from_numpy( np.asarray( [ train_labels[ 0 ] ] ) ).shape )
loss = model.evalModel( torch.unsqueeze( train_data[ 0, :, :, : ], 0 ), torch.from_numpy( np.asarray( [ train_labels[ 0 ] ] )  ).type(torch.cuda.LongTensor) )
print (loss)
'''

loss = model.evalModel( train_data[0:BATCH_SIZE, :, :, :], train_labels[0:BATCH_SIZE] )
print (loss)

#Gradient
'''
grad = model.computeGradientIter( torch.unsqueeze( train_data[ 0, :, :, : ], 0 ), torch.from_numpy( np.asarray( [ train_labels[ 9 ] ] )  ).type(torch.cuda.LongTensor), 0 )
'''
grad = model.computeGradientIter( train_data[0:BATCH_SIZE, :, :, :], train_labels[ 0:BATCH_SIZE ], 0 )
print( 'Gradient-----' )
print( len(grad) )

#for i in grad: 
#	print( i.data.cpu().numpy().flatten()[1:10] )

#HessianVec
#grad = torch.cat( [ g.contiguous ().view( -1 ) for g in grad if g is not None ] )
#vec = np.array( [ 0.1 for i in range( len( grad ) ) ] )
#vec = Variable( torch.from_numpy( vec ).type( torch.DoubleTensor ) )
vec = []
for c in grad: 
	w = torch.randn( c.size () ).type(torch.DoubleTensor) * 0.1
	vec.append( w.cuda () )

#hv = model.computeKFACHv( torch.unsqueeze( train_data[ 0, :, :, : ], 0 ), vec  )
hv = model.computeKFACHv( train_data[ 0:BATCH_SIZE, :, :, : ], vec )

print
print
print

for i in hv: 
	print( i.size () )
