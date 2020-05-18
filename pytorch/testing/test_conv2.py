
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import readWeights

import numpy as np

from swish_activation import Swish

DATASET_SIZE = 1

torch.set_printoptions( precision=10 )

def CONVHOOK( module, grad_input, grad_output ): 
	print('CONVHOOK -- > module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )

def ACTHOOK( module, grad_input, grad_output ): 
	print('ACTHOOK -- > module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )


class TestCNN(nn.Module):

	def __init__(self): 
		super(TestCNN, self).__init__()
		#self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
		#self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)

		self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=True)
		self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=0, bias=True)
		
		self.activation1 = nn.Softplus ()
		self.activation2 = nn.Softplus () 


		self.batch1 = nn.BatchNorm2d( 2, affine=False )
		self.batch2 = nn.BatchNorm2d( 3, affine=False )

		self.lossFunction = nn.CrossEntropyLoss()

		#self.conv1.register_backward_hook( CONVHOOK )
		#self.activation1.register_backward_hook( ACTHOOK )
		#self.conv2.register_backward_hook( CONVHOOK )
		#self.activation2.register_backward_hook( ACTHOOK )

	def setWeights( self, cw1, cb1, cw2, cb2 ): 
		self.conv1.weight.data.copy_( cw1 )
		self.conv1.bias.data.copy_( cb1 )

		self.conv2.weight.data.copy_( cw2 )
		self.conv2.bias.data.copy_( cb2 )

	def forward(self, x): 

		out = self.conv1( x )
		print( 'Output of first convolution... ')
		print (out[0, 0, :, :])

		out = self.activation1( out )
		print( 'Outpt of first activation... ')
		print (out)
		out = F.avg_pool2d( out, kernel_size=2, stride=2, padding=0)
		print( 'output of first pooling.... ')
		print (out)

		print( 'output of first batch n.... ')
		out = self.batch1( out )
		print( out )

		out = self.conv2( out )
		print( 'output of second convolution... ')
		print (out)
		out = self.activation2(out)
		print( 'output of second activation... ')
		print (out)
		out = F.avg_pool2d( out, kernel_size=2, stride=2, padding=0)
		print( 'output of second pooling... ')
		print (out)

		out = self.batch2( out )
		print( 'output of second batch ... ')
		print( out )

		out = out.view(out.size(0), -1) 
		print( 'output of the network... ')
		print (out)
		return out

	def evalModel( self, X, Y): 

		x_var = Variable( X ) 
		y_var = Variable( Y ) 

		out = self( x_var )
		loss = self.lossFunction( out, y_var )
		print( 'loss function...', loss )

		x_var.volatile=True
		y_var.volatile=True
		return loss

	def backwardPass( self, func, create_graph ): 
		grad = autograd.grad( func, self.parameters (), create_graph=create_graph)
		return grad


model = TestCNN ()
model.double ()

X = np.asarray( [1 for i in range( 1, (2*14*14 + 1) ) ] ) 
X = X.reshape( 1, 2, 14, 14 )
#X = np.tile( X, (1, 1, 1, 1) )

print( 'Reading Dataset from the matrix file.... ')
temp = readWeights.readMatrix( '../../cuda_dataset.txt', [ [DATASET_SIZE, 1, 14, 14] ] )
print( 'Done reading the Dataset from the file... ')
X = temp[ 0 ]

#X[ 0, 0, :, : ] = X[ 0, 0, :, : ].T
#X[ 0, 1, :, : ] = X[ 0, 1, :, : ].T

print( X[ 0, 0, :, : ] )
print
print


Y = np.asarray( DATASET_SIZE * [ 0 ] )

# Weights here. 
cw1 = np.array( [ 0.1 for i in range( 1, 4 * 2 * 3 * 3 + 1 ) ] )
cw1 = cw1.reshape( 4, 2, 3, 3 )
#cw1 = np.tile( cw1, (1, 1, 1, 1) )
cb1 = np.array( [ 0.1, 0.1, 0.1, 0.1  ] )

cw2 = np.array( [ 0.1 for i in range( 1, 6 * 4 * 3 * 3 + 1 ) ] )
cw2 = cw2.reshape( 6, 4, 3, 3 )
#cw2 = np.tile( cw2, (1, 1, 1, 1) )
cb2 = np.array( [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ] )

print( 'Reading Weights from the matrix file.... ')
temp = readWeights.readMatrix( '../../cuda_weights.txt', [ [2, 1, 3, 3], [2], [3, 2, 3, 3], [3] ] )
print( 'Done reading the Weights from the file... ')
cw1 = temp[  0 ]
cb1 = temp[ 1 ]
cw2 = temp[  2 ]
cb2 = temp[ 3 ]

print( cw1, cb1, cw2, cb2 )
model.setWeights( 
	torch.from_numpy( cw1 ).type( torch.DoubleTensor ), torch.from_numpy( cb1 ).type( torch.DoubleTensor ),
	torch.from_numpy( cw2 ).type( torch.DoubleTensor ), torch.from_numpy( cb2 ).type( torch.DoubleTensor ) )


#run the model here. 
ll = model.evalModel( torch.from_numpy( X ).type( torch.DoubleTensor ), torch.from_numpy( Y ).type( torch.LongTensor ) )


gradient = model.backwardPass( ll, True )

print
print
print

print( 'Gradient evaluation .... ')
print( gradient )
gradient = torch.cat( [ grad.contiguous ().view( -1 ) for grad in gradient if grad is not None ] )


# hessian vec is 
print
print
print ('Begin hessian vec....\n\n\n\n\n\n\n\n\n')

vec = np.array( [ 0.1 for i in range( len( gradient ) ) ] )

print( 'Reading Vector from the matrix file.... ')
#temp = readWeights.readMatrix( '../../cuda_weights2.txt', [ [1, 1, 5, 5], [1], [1, 1, 5, 5 ], [1] ] )
temp = readWeights.readMatrix( '../../cuda_weights2.txt', [ [2, 1, 3, 3], [2], [3, 2, 3, 3], [3] ] )
print( 'Done reading the Weights from the file... ')
a = temp[0]
b = temp[1]
c = temp[2]
d = temp[3]
a = np.reshape( a, (2, 1, 3, 3), order='F' )
a = np.ravel( a )
c = np.reshape( c, (3, 2, 3, 3), order='F' )
c = np.ravel( c )
vec = np.concatenate( (a, b, c, d) )


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

vec = Variable( torch.from_numpy( vec ).type( torch.DoubleTensor ) )


hv = model.backwardPass( (gradient * vec).sum (), False )

print ()
print ()
print ()

print( hv )