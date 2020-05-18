
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import readWeights

import numpy as np

class Swish(nn.Module):

	def forward(self, input):
		return input * torch.sigmoid(input)

	def __repr__(self):
		return self.__class__.__name__ + '()'

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
		self.conv1 = nn.Conv2d(1, 1, 3)
		self.activation = Swish ()

		self.lossFunction = nn.CrossEntropyLoss()

		self.conv1.register_backward_hook( CONVHOOK )
		self.activation.register_backward_hook( ACTHOOK )

	def setConvWeights( self, cw, cb ): 
		self.conv1.weight.data.copy_( cw )
		self.conv1.bias.data.copy_( cb )

	def forward(self, x): 
		out = self.conv1( x )
		print( 'output of convolutoin', out )
		out = self.activation(out)
		print( 'output of c-activation', out )
		out = F.avg_pool2d(out, 2)
		print( 'output of pooling', out )
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


model = TestCNN ()
model.double ()

X = np.asarray( [1 for i in range( 1, 37 ) ] ) 
X = X.reshape( 1, 1, 6, 6 )
X = np.tile( X, (1, 1, 1, 1) )

print( 'Reading Dataset from the matrix file.... ')
temp = readWeights.readMatrix( '../../cuda_dataset.txt', [ [1, 1, 6, 6] ] )
print( 'Done reading the Dataset from the file... ')
X = temp[ 0 ]

print( X )
print
print


Y = np.asarray( [ 0 ] )

# Weights here. 
cw = np.array( [ 0.1 for i in range( 1, 10 ) ] )
cw = cw.reshape( 1, 1, 3, 3 )
cw = cw.T
cw = np.tile( cw, (1, 1, 1, 1) )
cb = np.array( [ 0.1 ] )

print( 'Reading Weights from the matrix file.... ')
temp = readWeights.readMatrix( '../../cuda_weights.txt', [ [1, 1, 3, 3], [1] ] )
print( 'Done reading the Weights from the file... ')
cw = temp[  0 ]
cb = temp[ 1 ]

print( cw, cb )
model.setConvWeights( torch.from_numpy( cw ).type( torch.DoubleTensor ), torch.from_numpy( cb ).type( torch.DoubleTensor ) )


#run the model here. 
ll = model.evalModel( torch.from_numpy( X ).type( torch.DoubleTensor ), torch.from_numpy( Y ).type( torch.LongTensor ) )

print
print
print

print( 'Gradient evaluation .... ')
gradient = model.backwardPass( ll, True )
print( gradient )
gradient = torch.cat( [ grad.contiguous ().view( -1 ) for grad in gradient if grad is not None ] )

				
# hessian vec is 
print
print
print ('Begin hessian vec....')
#print( gradient )

vec = np.array( [ 0.1 for i in range( len( gradient ) ) ] )


print( 'Reading Vector from the matrix file.... ')
temp = readWeights.readMatrix( '../../cuda_weights2.txt', [ [1, 1, 3,3], [1] ] )
print( 'Done reading the Weights from the file... ')
c = temp[0]
b = temp[1]

c  = np.array( c )
c = np.reshape( c, (1, 1, 3, 3), order='F' )
c = np.ravel( c )
b = np.array( b )

vec = np.concatenate( (c, b) )


#cw = np.array( [ 0.1*i for i in range( 1, 10 ) ] )
#vec = np.concatenate( [ cw, cb, [0.1*i for i in range(1, 9)], lb ] )



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

print
print
print

print( hv )

