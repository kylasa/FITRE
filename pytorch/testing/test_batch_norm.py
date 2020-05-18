
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import readWeights

import numpy as np

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
		self.conv1 = nn.Conv2d(2, 2, 2)
		self.batch1 = nn.BatchNorm2d( 2, affine=False)

		self.conv1.register_backward_hook( CONVHOOK )
		self.batch1.register_backward_hook( ACTHOOK )

		self.lossFunction = nn.MSELoss()

	def setConvWeights( self, cw, cb ): 
		self.conv1.weight.data.copy_( cw )
		self.conv1.bias.data.copy_( cb )

	def forward(self, x): 

		out = self.conv1( x )
		self.batch_data =  out.clone ()
		print( 'output of the convolution', out )

		out = self.batch1( out )
		print( 'output of batch normalization', out )

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


X = np.asarray( [i for i in range( 1, 19 ) ] ) 
X = X.reshape( 1, 2, 3, 3 )
X = np.tile( X, (1, 1, 1, 1) )


# Weights here. 
cw = np.array( [ 1 for i in range( 1, 17 ) ] )
cw = cw.reshape( 2, 2, 2, 2 )
cb = np.array( [ 0 ] )
print ()
print ()
print ('Weights and Biases')
print( cw, cb )

model.setConvWeights( torch.from_numpy( cw ).type( torch.DoubleTensor ), torch.from_numpy( cb ).type( torch.DoubleTensor ) )


print( X )
print
print


Y = np.asarray( [ 1, 0, 0, 0, 0, 0, 0, 0 ] )

#run the model here. 
ll = model.evalModel( torch.from_numpy( X ).type( torch.DoubleTensor ), torch.from_numpy( Y ).type( torch.DoubleTensor ) )
gradient = model.backwardPass( ll, True )

print ()
print ()
print ( model.batch1.running_mean )
print ( model.batch1.running_var )
print ()
print ()
print ('Gradient --> ' )
print (gradient)

print( 'Batch norm Data --> ')
BATCH_DATA =  model.batch_data.data.numpy ()
print( BATCH_DATA.shape )
print( BATCH_DATA )
print( np.mean( BATCH_DATA[0, 0, :, :] ) )
print( np.mean( BATCH_DATA[0, 1, :, :] ) )
print( np.var( BATCH_DATA[0, 0, :, :] ) )
print( np.var( BATCH_DATA[0, 1, :, :] ) )

exit ()

