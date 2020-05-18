
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import readWeights
from lenet_cifar import LeNetCIFAR

import numpy as np

BATCH_SIZE = 1

model = LeNetCIFAR( num_classes = 10, activation='softplus')
model.double ()
model.cuda ()
criterion = nn.CrossEntropyLoss ()
#criterion = nn.CrossEntropyLoss (size_average=False)
model.setLossFunction( criterion )

X = np.asarray( [1 for i in range( 1, (3*32*32 + 1) ) ] ) 
X = X.reshape( 1, 3, 32, 32 )
#X = np.tile( X, (1, 1, 1, 1) )

print( 'Reading Dataset from the matrix file.... ')
temp = readWeights.readMatrix( '../../cuda_dataset.txt', [ [BATCH_SIZE, 3, 32, 32] ] )
print( 'Done reading the Dataset from the file... ')
#X = temp[ 0 ]
X = np.reshape( temp[0], (BATCH_SIZE, 3, 32, 32), order='F' )

print( X )
print
print


Y = np.asarray( [ 0 for i in range(BATCH_SIZE) ] )

# Weights here. 
print( 'Reading Weights from the matrix file.... ')
temp = readWeights.readMatrix( '../../cuda_weights.txt', [ [6, 3, 5, 5], [6], [16, 6, 5, 5], [16], [120, 400], [120], [84, 120], [84], [10, 84], [10] ] )
print( 'Done reading the Weights from the file... ')
cw1 = temp[  0 ]
cw1 = np.reshape( cw1, (6, 3, 5, 5), order='F' )
cb1 = temp[ 1 ]
cw2 = temp[  2 ]
cw2 = np.reshape( cw2, (16, 6, 5, 5), order='F' )
cb2 = temp[ 3 ]

lw1 = temp[ 4 ]
lw1 = np.reshape( lw1, (120, 400), order='F' )
lb1 = temp[ 5 ]
lw2 = temp[ 6 ]
lw2 = np.reshape( lw2, (84, 120), order='F' )
lb2 = temp[ 7 ]
lw3 = temp[ 8 ]
lw3 = np.reshape( lw3, (10, 84), order='F' )
lb3 = temp[ 9 ]

#print( cw1, cb1, cw2, cb2, lw1, lb1, lw2, lb2, lw3, lb3 )
model.setWeightsBiases( 
	torch.from_numpy( cw1 ).type( torch.cuda.DoubleTensor ), torch.from_numpy( cb1 ).type( torch.cuda.DoubleTensor ),
	torch.from_numpy( cw2 ).type( torch.cuda.DoubleTensor ), torch.from_numpy( cb2 ).type( torch.cuda.DoubleTensor ), 
	torch.from_numpy( lw1 ).type( torch.cuda.DoubleTensor ), torch.from_numpy( lb1 ).type( torch.cuda.DoubleTensor ), 
	torch.from_numpy( lw2 ).type( torch.cuda.DoubleTensor ), torch.from_numpy( lb2 ).type( torch.cuda.DoubleTensor ), 
	torch.from_numpy( lw3 ).type( torch.cuda.DoubleTensor ), torch.from_numpy( lb3 ).type( torch.cuda.DoubleTensor )
 )

#model.initZeroWeights ()

#run the model here. 
ll = model.evalModel( torch.from_numpy( X ).type( torch.cuda.DoubleTensor ), torch.from_numpy( Y ).type( torch.cuda.LongTensor ) )
print( 'Done evaluating the model')
print( ll )
#gradient = model.backwardPass( ll, True )
gradient = model._computeGradient( ll, True)
print( 'Done evaluating the gradient' )

print
print
print

print( 'Gradient evaluation .... ')
print( gradient )
'''
gradient = torch.cat( [ grad.contiguous ().view( -1 ) for grad in gradient if grad is not None ] )


# hessian vec is 
print
print
print ('Begin hessian vec....')
#print( gradient )

vec = np.array( [ 0.1 for i in range( len( gradient ) ) ] )

print( 'Reading Vector from the matrix file.... ')
temp = readWeights.readMatrix( '../../cuda_weights2.txt', [ [6, 3, 5, 5], [6], [16, 6, 5, 5], [16], [120, 400], [120], [84, 120], [84], [10, 84], [10] ] )
print( 'Done reading the Weights from the file... ')
cw1 = temp[  0 ]
cb1 = temp[ 1 ]
cw2 = temp[  2 ]
cb2 = temp[ 3 ]

lw1 = temp[ 4 ]
lb1 = temp[ 5 ]
lw2 = temp[ 6 ]
lb2 = temp[ 7 ]
lw3 = temp[ 8 ]
lb3 = temp[ 9 ]

cw1 = np.reshape( cw1, (6, 3, 5, 5), order='F' )
cw1 = np.ravel( cw1 )
cw2 = np.reshape( cw2, (16, 6, 5, 5), order='F' )
cw2 = np.ravel( cw2 )
lw1 = np.reshape( lw1, (120, 400), order='F' )
lw1 = np.ravel( lw1 )
lw2 = np.reshape( lw2, (84, 120), order='F' )
lw2 = np.ravel( lw2 )
lw3 = np.reshape( lw3, (10, 84), order='F' )
lw3 = np.ravel( lw3 )

vec = np.concatenate( (cw1, cb1, cw2, cb2, lw1, lb1, lw2, lb2, lw3, lb3) )

print ('This is the vector... ')
print
print
print
print
print
print (vec )

print
print
print

vec = Variable( torch.from_numpy( vec ).type( torch.cuda.DoubleTensor ) )


#hv = model.backwardPass( (gradient * vec).sum (), False )
hv = model._computeGradient( (gradient * vec).sum (), False )

print
print
print

print( hv )
'''
