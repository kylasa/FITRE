
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

from swish_activation import Swish

import readWeights

import numpy as np

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

def POOLHOOK( module, grad_input, grad_output ): 
	print('POOLHOOK -- > module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )

def FOR_CONVHOOK( module, grad_input, grad_output ): 
	print('FOR_CONVHOOK -- > module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )

def FOR_ACTHOOK( module, grad_input, grad_output ): 
	print('FOR_ACTHOOK -- > module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )

def BN_HOOK( module, grad_input, grad_output ): 
	print( 'batch norm grad_input', grad_input )
	print( 'batch norm grad_output', grad_output )



class TestCNN(nn.Module):

	def __init__(self): 
		super(TestCNN, self).__init__()
		#self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=1, padding=2)
		self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=0)
		self.activation = nn.Softplus ()
		#self.activation = Swish ()
		#self.pool = nn.functional.avg_pool2d ()
		#self.pool = nn.MaxPool2d( kernel_size=3, stride=2, padding=1 )
		self.pool = nn.MaxPool2d( kernel_size=2 )
		self.bn = nn.BatchNorm2d( 4, affine=False );	

		self.lossFunction = nn.CrossEntropyLoss()

		self.conv1.register_backward_hook( CONVHOOK )
		#self.conv1.register_forward_hook( FOR_CONVHOOK )
		self.activation.register_backward_hook( ACTHOOK )
		#self.activation.register_forward_hook( FOR_ACTHOOK )
		self.pool.register_backward_hook( POOLHOOK )
		#self.pool.register_forward_hook( POOLHOOK )

		self.bn.register_backward_hook( BN_HOOK )

	def setConvWeights( self, cw, cb ): 
		self.conv1.weight.data.copy_( cw )
		self.conv1.bias.data.copy_( cb )

	def forward(self, x): 
		out = self.conv1( x )
		print( 'output of convolutoin', out )
		out = self.activation(out)
		print( 'output of c-activation', out )
		#out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
		#out = F.max_pool2d(out, kernel_size=2 )
		out = self.pool( out )
		print( 'output of pool', out )
		out = self.bn( out )
		print( 'output of bn', out )
		#out = out.transpose_( 3, 2).contiguous ()
		out = out.view(out.size(0), -1) 
		print( 'output of the forward pass', out )
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
		g = autograd.grad( func, self.parameters (), create_graph=create_graph)
		return g
		#return torch.cat( [ grad.contiguous ().view( -1 ) for grad in g if grad is not None ] ) 

	def hv( self, fun, grad, vec ): 
		hv = autograd.grad( grad, self.parameters (), grad_outputs=vec, retain_graph=True, only_inputs=True )
		return hv


model = TestCNN ()
model.double ()

#X = np.asarray( [1 for i in range( 1, 73 ) ] ) 
#X = X.reshape( 1, 2, 6, 6 )
#X = np.tile( X, (1, 1, 1, 1) )

print( 'Reading Dataset from the matrix file.... ')
temp = readWeights.readMatrix( '../../cuda_dataset.txt', [ [DATASET_SIZE, 2, 6, 6] ] )
print( 'Done reading the Dataset from the file... ')

X = temp[ 0 ]

print( X )
print
print


Y = np.asarray( DATASET_SIZE * [ 0 ] )

# Weights here. 
cw = np.array( [ 0.1 for i in range( 1, 73 ) ] )
cw = cw.reshape( 4, 2, 3, 3 )
cw = cw.T
cw = np.tile( cw, (1, 1, 1, 1) )
cb = np.array( [ 0.1, 0.1, 0.1, 0.1 ] )

print( 'Reading Weights from the matrix file.... ')
temp = readWeights.readMatrix( '../../cuda_weights.txt', [ [4, 2, 3, 3], [4] ] )
print( 'Done reading the Weights from the file... ')
cw = temp[  0 ]
cb = temp[ 1 ]

print( cw, cb )
model.setConvWeights( torch.from_numpy( cw ).type( torch.DoubleTensor ), torch.from_numpy( cb ).type( torch.DoubleTensor ) )


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
print ('Begin hessian vec....')
#print( gradient )

vec = np.array( [ 0.1 for i in range( len( gradient ) ) ] )


print( 'Reading Vector from the matrix file.... ')
temp = readWeights.readMatrix( '../../cuda_weights2.txt', [ [4, 2, 3,3], [4] ] )
print( 'Done reading the Weights from the file... ')
c = temp[0]
b = temp[1]

print( c )
print( b )


#cw = np.array( [ 0.1*i for i in range( 1, 10 ) ] )
#vec = np.concatenate( [ cw, cb, [0.1*i for i in range(1, 9)], lb ] )

c  = np.array( c )
c = np.reshape( c, (4, 2, 3, 3), order='F' )
c = np.ravel( c )
b = np.array( b )

vec = np.concatenate( (c, b) )


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
#hv = model.hv( ll, gradient, vec )

print ()
print ()
print ()

print( hv )

