
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import readWeights
import numpy as np

def FCHOOK( module, grad_input, grad_output ): 
	print('FC -- > module hook')
	print( 'grad in', grad_input )
	print( 'grad out', grad_output )

class TestCNN(nn.Module):

	def __init__(self): 
		super(TestCNN, self).__init__()
		self.fc1   = nn.Linear(2*2, 2)
		self.lossFunction = nn.CrossEntropyLoss()

		self.fc1.register_backward_hook( FCHOOK )

	def setLinearWeights( self, lw, lb ): 
		self.fc1.weight.data.copy_( lw )
		self.fc1.bias.data.copy_( lb )

	def forward(self, x): 

		out = self.fc1( x )
		print( 'output of wz + b', out )
		out = F.softplus(out)
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

	def printGradient( self ): 
		print( self.conv1.weight.grad)
		print( self.fc1.bias.grad)


model = TestCNN ()
model.double ()

X = np.asarray( [1 for i in range( 1, 5 ) ] ) 
X = X.reshape( (1, 4) )
print( X )
print
print

Y = np.asarray( [ 0 ] )

# Weights here. 
lw = np.array( [[ 0.1, 0.2, 0.3, 0.4 ], [0.5, 0.6, 0.7, 0.8]] )
lb = np.array( [ 0.1, 0.2] )
'''
lw = np.array( [ 0.1 for i in range( 1, 6 * 2 * 2 * 10 + 1 ) ] )
lw = lw.reshape( 10, 24 )
lb = np.array( [ 0.1 for i in range( 1, 11 ) ] )
'''


print( lw, lb )
model.setLinearWeights( torch.from_numpy( lw ).type( torch.DoubleTensor ), torch.from_numpy( lb ).type( torch.DoubleTensor ) )


#run the model here. 
ll = model.evalModel( torch.from_numpy( X ).type( torch.DoubleTensor ), torch.from_numpy( Y ).type( torch.LongTensor ) )
gradient = model.backwardPass( ll, True )

print
print
print

print( 'Gradient evaluation .... ')



# hessian vec is 
print
print
print ('Begin hessian vec....')
print( gradient )
gradient = torch.cat( [ grad.contiguous ().view( -1 ) for grad in gradient if grad is not None ] )


vec = np.array( [ 0.1 for i in range( len( gradient ) ) ] )
vec = Variable( torch.from_numpy( vec ).type( torch.DoubleTensor ) )
hv = model.backwardPass( (gradient * vec).sum (), False )

print
print
print

print( hv )

