
import torch
import torch.nn
from torch.autograd import Variable
from torch import autograd
import numpy as np
import readWeights


from swish_activation import Swish

def CONVHOOK( module, grad_input, grad_output ):  
   print('CONVHOOK -- > module hook')
   print( 'grad input', grad_input )
   print( 'grad out', grad_output )

def POOLHOOK( module, grad_input, grad_output ):  
   print('POOLHOOK -- > module hook')
   print( 'grad input', grad_input )
   print( 'grad out', grad_output )

X = np.asarray( [ 1 for i in range( 1, 37 ) ] )
#X = np.random.randn( 6, 6)
X = X.reshape( 6, 6 )
X = np.tile( X, ( 1, 1, 1, 1) )

temp = readWeights.readMatrix( '../../cuda_dataset.txt', [ [1, 1, 6, 6] ] )
#X = np.reshape( temp[0], (1, 1, 6, 6), order='F' )
X = temp[0]



#data = torch.from_numpy( np.asarray( [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ], dtype=float ) )
#data = torch.unsqueeze( data, 0 )
#data = Variable( data, requires_grad=True )

data = Variable( torch.from_numpy( X ).type(torch.DoubleTensor), requires_grad=True )
print(data)

#target = torch.from_numpy( np.asarray( [ 0, 0, 0, 1 ] ) ).type( torch.FloatTensor )
target = torch.from_numpy( np.asarray( [ 0 ] ) )
target = Variable( target )

l1 = torch.nn.Linear( 4, 2 )
l1.weight.data.fill_(0)
l1.bias.data.fill_(0)


p1 = torch.nn.MaxPool2d( 2 )
#p1 = torch.nn.AvgPool2d( 2 )
p1.double ()

c1 = torch.nn.Conv2d (1, 1, 3)
c1.weight.data.fill_(np.float64(0.1))
c1.bias.data.fill_(np.float64(0.1))
c1.double ()

temp = readWeights.readMatrix( '../../cuda_weights.txt', [ [1, 1, 3, 3], [1] ] )
#cw1 = np.reshape( temp[0], (1, 1, 3, 3), order='F' )
c1.weight.data.copy_ (torch.from_numpy( temp[0] ).type( torch.DoubleTensor ))

cb1 = temp[ 1 ] 
c1.bias.data.copy_(torch.from_numpy( cb1).type( torch.DoubleTensor ))

activation = Swish ()
activation.double ()



c1.register_backward_hook( CONVHOOK )
p1.register_backward_hook( POOLHOOK )

loss = torch.nn.CrossEntropyLoss ()
#loss = torch.nn.MSELoss ()
out = c1( data )
#out = l1( out )
print( 'output of convolution: ', out )
out = torch.nn.functional.softplus( out )
#out = activation( out )
print( 'output of softplus: ', out )
#out = torch.nn.functional.max_pool2d( out, 2 )
out = p1( out )
print( 'output of the pooling layer. ', out )

lsVal =  loss( out.view( out.size(0), -1), target ) 
print( out )
print( lsVal )

#lsVal.backward ()
#exit ()

g = autograd.grad( lsVal, list(c1.parameters ()), create_graph=True )
gradient = torch.cat( [ grad.contiguous ().view( -1 ) for grad in g if grad is not None ] )

print( 'gradient is ', g )

vec = np.asarray( [ 0.1 for i in range( 1, len(gradient) + 1 ) ] )
vector = Variable( torch.from_numpy( vec ).type(torch.DoubleTensor) )

temp = readWeights.readMatrix( '../../cuda_weights2.txt', [ [1, 1, 3, 3], [1] ] )
#cw1 = np.ravel( np.reshape( temp[0],(1, 1, 3, 3), order='F'  ))
cw1 = np.ravel(temp[0] )
cb1 = temp[1]

#cw1 = np.asarray( [ 0.8647294994, 0.8964337872, 0.1227971409, 0.8949059114, 0.9372991756, 0.0170291554, 0.5209090252, 0.0900310910, 0.4445049161 ] )
#cw1 = np.asarray( [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ] )
#cb1 = np.asarray( [ 0.1 ] )

#cw1[0] = 1
#cw1[1:] = 0
#cb1[0] = 0

vec = np.concatenate( (cw1, cb1) )
vector = Variable( torch.from_numpy( vec ).type(torch.DoubleTensor) )
print( cw1 )
print( cb1 )

print( 'Hessian Vector Being.... ')


gv = (vector * gradient).sum ()
hv = autograd.grad( gv, list( c1.parameters () ), create_graph=False )

print( 'Hessian Vector is : ', hv )
