
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

def CONVHOOK( module, grad_input, grad_output ): 
	print('module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )


def FCHOOK( module, grad_input, grad_output ): 
	print('module hook')
	print( 'grad out', grad_output )

class TestCNN(nn.Module):

	def __init__(self): 
		super(TestCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 3)
		self.fc1   = nn.Linear(6*2*2, 10)
		self.lossFunction = nn.CrossEntropyLoss()

		self.conv1.register_backward_hook( CONVHOOK )
		#self.fc1.register_backward_hook( FCHOOK )

	def setConvWeights( self, cw, cb ): 
		#self.conv1.weight.data.copy_( cw )
		#self.conv1.bias.data.copy_( cb )
		self.conv1.weight.data.fill_ (0) 
		self.conv1.bias.data.fill_ (0)

	def setLinearWeights( self, lw, lb ): 
		#self.fc1.weight.data.copy_( lw )
		#self.fc1.bias.data.copy_( lb )
		self.fc1.weight.data.fill_ (1)
		self.fc1.bias.data.fill_ (1)

	def forward(self, x): 
		out = self.conv1( x )
		#print( 'After Convolution: .... ')
		#print( out )
		out = F.softplus(out)
		#print( 'After softplus.... ')
		#print(out)
		out = F.avg_pool2d(out, 2)
		#print( 'After Pooling ... ')
		#jprint(out)
		out = out.view(out.size(0), -1) 
		out = self.fc1( out )
		#print( 'After Linear Layer... ')
		#print(out)
		out = F.softplus(out)
		#print( 'After Linear Activation ... ')
		#print(out)
		return out

	def evalModel( self, X, Y): 
		x_var = Variable( X ) 
		y_var = Variable( Y ) 

		out = self( x_var )
		loss = self.lossFunction( out, y_var )

		x_var.volatile=True
		y_var.volatile=True
		return loss

	def backwardPass( self, func, create_graph ): 
		g = autograd.grad( func, self.parameters (), create_graph=create_graph)
		return torch.cat( [ grad.contiguous ().view( -1 ) for grad in g if grad is not None ] ) 


model = TestCNN ()

X = np.asarray( [ i for i in range( 1, 37 ) ] ) 
X = X.reshape( 6, 6 ) #+ 9 
X = np.tile( X, (1, 3, 1, 1) )

#Y = np.asarray( [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] )
Y = np.asarray( [ 1 ] )


w = np.asarray( [i for i in range( 1,  6 * 3 * 3 * 3 + 1 )]) * 0.01
w = w.reshape( (6, 3, 3, 3) )

b = np.asarray( [ i for i in range( 1, 7) ] ) * 0.1 

lw = np.asarray( [ i for i in range( 1, 6 * 2 * 2 * 10 + 1 ) ] )
lw = lw.reshape( ( 10, 6 * 2 * 2 ) )
lb = np.asarray( [ i for i in range( 1, 11) ] ) 

model.setConvWeights ( torch.from_numpy( w ), torch.from_numpy( b ) )
model.setLinearWeights ( torch.from_numpy( lw ), torch.from_numpy( lb ) )


#run the model here. 
ll = model.evalModel( torch.from_numpy( X ).type( torch.FloatTensor ), torch.from_numpy( Y ).type( torch.LongTensor ) )
gradient = model.backwardPass( ll, True )

#print( 'Gradient is as follows: ' )
#print( gradient )
