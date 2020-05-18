
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import readWeights

import numpy as np

def CONVHOOK( module, grad_input, grad_output ): 
	print( 'Module -->', module )
	print( 'Module -->', module.in_channels )
	print('CONVHOOK -- > module hook')
	print( 'grad input', grad_input )
	print( 'grad out', grad_output )


def FCHOOK( module, grad_input, grad_output ): 
	print( 'Module -->', module )
	print('FC -- > module hook')
	print( 'grad in', grad_input )
	print( 'grad out', grad_output )

class TestCNN(nn.Module):

	def __init__(self): 
		super(TestCNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 1, 3)
		self.fc1   = nn.Linear(1*2*2, 2)
		self.lossFunction = nn.CrossEntropyLoss()

		self.conv1.register_backward_hook( CONVHOOK )
		self.fc1.register_backward_hook( FCHOOK )

	def setConvWeights( self, cw, cb ): 
		self.conv1.weight.data.copy_( cw )
		self.conv1.bias.data.copy_( cb )
		#self.conv1.weight.data.fill_(1) 
		#self.conv1.bias.data.fill_ (1)

	def setLinearWeights( self, lw, lb ): 
		self.fc1.weight.data.copy_( lw )
		self.fc1.bias.data.copy_( lb )
		#self.fc1.weight.data.fill_ (1)
		#self.fc1.bias.data.fill_ (1)

	def forward(self, x): 
		out = self.conv1( x )
		print( 'output of convolutoin', out )
		out = F.softplus(out)
		print( 'output of c-activation', out )
		out = F.avg_pool2d(out, 2)
		print( 'output of pool', out )

		out = out.transpose_( 3, 2).contiguous ()
		print( 'output of pool', out )
		convOut = out.view(out.size(0), -1) 

		out = self.fc1( convOut )
		#print( 'output of WZ + b', out )
		#print( 'output of WZ + b, in CUDA format is',  np.add( np.matmul( self.fc1.weight.data, convOut.data.numpy().reshape( 2, 2).T.reshape(4, 1)), self.fc1.bias.data.numpy ()) )
		out = F.softplus(out)
		print( 'output of the forward pass', out )
		return out

	def evalModel( self, X, Y): 

		#print( self.conv1.weight.data )
		#print( self.conv1.bias.data )

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
#model.setConvWeights( None, None)
#model.setLinearWeights( None, None )

X = np.asarray( [1 for i in range( 1, 37 ) ] ) 
X = X.reshape( 6, 6 )
X = X.T
X = np.tile( X, (1, 1, 1, 1) )
print( X )
print
print

Y = np.asarray( [ 0 ] )

# Weights here. 
cw = np.array( [ 0.1 for i in range( 1, 10 ) ] )
cw = cw.reshape( 3, 3 )
cw = cw.T
cw = np.tile( cw, (1, 1, 1, 1) )

cb = np.array( [ 0.1 ] )
lw = np.array( [[ 0.1, 0.1, 0.1, 0.1 ], [0.1, 0.1, 0.1, 0.1]] )
lb = np.array( [ 0.1, 0.1] )
'''
lw = np.array( [ 0.1 for i in range( 1, 6 * 2 * 2 * 10 + 1 ) ] )
lw = lw.reshape( 10, 24 )
lb = np.array( [ 0.1 for i in range( 1, 11 ) ] )
'''


print( cw, cb, lw, lb )
model.setConvWeights( torch.from_numpy( cw ).type( torch.DoubleTensor ), torch.from_numpy( cb ).type( torch.DoubleTensor ) )
model.setLinearWeights( torch.from_numpy( lw ).type( torch.DoubleTensor ), torch.from_numpy( lb ).type( torch.DoubleTensor ) )

'''
print( 'Reading Weights from the matrix file.... ')
temp = readWeights.readMatrix( '../cuda_weights.txt', [ [1, 1, 3, 3], [1], [2, 4], [2] ] )
print( 'Done reading the weights from the file... ')

result = []
for it in temp: 
	result.append( np.asfortranarray( it ) )

print( temp )
print( result )
model.setConvWeights( torch.from_numpy( result[0] ).type( torch.DoubleTensor ), torch.from_numpy( result[1] ).type( torch.DoubleTensor ) )
model.setLinearWeights( torch.from_numpy( result[2] ).type( torch.DoubleTensor ), torch.from_numpy( result[3] ).type( torch.DoubleTensor ) )
'''




#run the model here. 
ll = model.evalModel( torch.from_numpy( X ).type( torch.DoubleTensor ), torch.from_numpy( Y ).type( torch.LongTensor ) )
gradient = model.backwardPass( ll, True )

print
print
print

print( 'Gradient evaluation .... ')


'''
# hessian vec is 
print
print
print ('Begin hessian vec....')
print( gradient )
gradient = torch.cat( [ grad.contiguous ().view( -1 ) for grad in gradient if grad is not None ] )
print( gradient )


vec = np.array( [ 0.1 for i in range( len( gradient ) ) ] )
#cw = np.array( [ 0.1*i for i in range( 1, 10 ) ] )
#vec = np.concatenate( [ cw, cb, [0.1*i for i in range(1, 9)], lb ] )


vec = Variable( torch.from_numpy( vec ).type( torch.DoubleTensor ) )


hv = model.backwardPass( (gradient * vec).sum (), False )

print
print
print

print( hv )

'''
