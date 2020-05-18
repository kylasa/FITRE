
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

from swish_activation import Swish

import readWeights

import numpy as np

DATASET_SIZE = 4

torch.set_printoptions( precision=10 )

def CONVHOOK( module, grad_input, grad_output ): 
	print('CONVHOOK -- > module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )

def ACTHOOK( module, grad_input, grad_output ): 
	print('ACTHOOK -- > module hook')
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

def LINHOOK( module, grad_input, grad_output ): 
	print('LINHOOK -- > module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )

def LINACTHOOK( module, grad_input, grad_output ): 
	print('LINACTHOOK -- > module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )

def FOR_LINHOOK( module, grad_input, grad_output ): 
	print('FOR_LINHOOK -- > module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )

def FOR_LINACTHOOK( module, grad_input, grad_output ): 
	print('FOR_LINACTHOOK -- > module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )




class TestCNN(nn.Module):

	def __init__(self): 
		super(TestCNN, self).__init__()
		self.conv1 = nn.Conv2d(2, 4, 3, stride=1, padding=1, bias=False)
		self.activation = Swish ()
		self.bn = nn.BatchNorm2d( 4, affine=False )

		self.fc1 = nn.Linear( 4 * 8 * 8, 10, bias=False )
		#self.factivation = nn.Softplus()
		self.factivation = Swish ()
		self.pool = nn.MaxPool2d (kernel_size=2)

		self.lossFunction = nn.CrossEntropyLoss()

		#self.conv1.register_backward_hook( CONVHOOK )
		#self.conv1.register_forward_hook( FOR_CONVHOOK )
		#self.activation.register_backward_hook( ACTHOOK )
		#self.activation.register_forward_hook( FOR_ACTHOOK )
		#self.pool.register_backward_hook( CONVHOOK )

		#self.fc1.register_backward_hook( LINHOOK )
		#self.fc1.register_forward_hook( FOR_LINHOOK )

		#self.factivation.register_backward_hook( LINACTHOOK )
		#self.factivation.register_forward_hook( FOR_LINACTHOOK )

	def setConvWeights( self, cw, cb, cw1, cb1 ): 
		self.conv1.weight.data.copy_( cw )
		#self.conv1.bias.data.copy_( cb )

		self.fc1.weight.data.copy_( cw1 )
		#self.fc1.bias.data.copy_( cb1 )

	def forward(self, x): 
		out = self.conv1( x )
		print( 'output of convolutoin', out )
		out = self.activation(out)
		print( 'output of c-activation', out )
		out = self.pool( out ) #F.max_pool2d(out, 2)
		print( 'output of pool', out )

		#out = self.bn( out )
		#print( 'output of batch norm', out )
		#out = out.transpose_( 3, 2).contiguous ()
		#out = out.transpose( 2, 3 ).contiguous ()
		out = out.view(out.size(0), -1)
		#print( 'input to the linear layer: ', out )

		out = self.fc1( out )
		print( 'output of linear layer: ', out )
		out = self.factivation( out )
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
temp = readWeights.readMatrix( '../../cuda_dataset.txt', [ [DATASET_SIZE, 2, 16, 16] ] )
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
#temp = readWeights.readMatrix( '../../cuda_weights.txt', [ [4, 2, 3, 3], [4], [4, 16], [4] ] )
temp = readWeights.readMatrix( '../../cuda_weights.txt', [ [4, 2, 3, 3], [10, 4 * 8 * 8] ] )
print( 'Done reading the Weights from the file... ')
cw = temp[  0 ]
cl = temp[ 1 ]

cb = None
clb = None

print( cw, cb, cl, clb )
#model.setConvWeights( torch.from_numpy( cw ).type( torch.DoubleTensor ), torch.from_numpy( cb ).type( torch.DoubleTensor ), torch.from_numpy( cl ).type( torch.DoubleTensor ), torch.from_numpy( clb ).type( torch.DoubleTensor ) )
model.setConvWeights( torch.from_numpy( cw ).type( torch.DoubleTensor ), None, torch.from_numpy( cl ).type( torch.DoubleTensor ), None )


#run the model here. 
ll = model.evalModel( torch.from_numpy( X ).type( torch.DoubleTensor ), torch.from_numpy( Y ).type( torch.LongTensor ) )
print( 'Model Likelihood... ->', ll )


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
#temp = readWeights.readMatrix( '../../cuda_weights2.txt', [ [4, 2, 3,3], [4], [4, 16], [4] ] )
temp = readWeights.readMatrix( '../../cuda_weights2.txt', [ [4, 2, 3, 3], [10, 4 * 8 * 8] ] )
print( 'Done reading the Weights from the file... ')
c = temp[0]
b = None
c1 = temp[1]
b1 = None

print( c )
print( b )
print( c1 )
print( b1 )


#cw = np.array( [ 0.1*i for i in range( 1, 10 ) ] )
#vec = np.concatenate( [ cw, cb, [0.1*i for i in range(1, 9)], lb ] )

c  = np.array( c )
c = np.reshape( c, (4, 2, 3, 3), order='F' )
c = np.ravel( c )
b = np.array( b )

c1 = np.array( c1 )
c1 = np.reshape( c1, (10, 4 * 8 * 8), order='F' )
c1 = np.ravel( c1 )
b1 = np.array( b1 )

vec = np.concatenate( (c, c1) )

#vec = np.concatenate( (c, b, c1, b1) )


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

print
print
print

print( hv )
