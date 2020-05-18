
import torch
import torch.nn
from torch.autograd import Variable
from torch import autograd
import numpy as np

def CONVHOOK( module, grad_input, grad_output ):  
   print('CONVHOOK -- > module hook')
   print( 'grad input', grad_input )
   print( 'grad out', grad_output )

X = np.asarray( [ i for i in range( 1, 37 ) ] )
#X = np.random.randn( 6, 6)
X = X.reshape( 6, 6 )
X = np.tile( X, ( 1, 1, 1, 1) )


#data = torch.from_numpy( np.asarray( [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ], dtype=float ) )
#data = torch.unsqueeze( data, 0 )
#data = Variable( data, requires_grad=True )

data = Variable( torch.from_numpy( X ).type(torch.FloatTensor), requires_grad=True )
print(data)

target = torch.from_numpy( np.asarray( [ 1 ] ) )
target = Variable( target )

l1 = torch.nn.Linear( 4, 2 )
l1.weight.data.fill_(0)
l1.bias.data.fill_(0)

c1 = torch.nn.Conv2d (1, 1, 3)
c1.weight.data.fill_(1)
c1.bias.data.fill_(1)

c1.register_backward_hook( CONVHOOK )

loss = torch.nn.CrossEntropyLoss ()
out = c1( data )
out = l1( out )
print( 'output of convolution: ', out )
out = torch.nn.functional.softplus( out )
print( 'output of softplus: ', out )
out = torch.nn.functional.avg_pool2d( out, 2 )
print( 'output of the pooling layer. ', out )

lsVal =  loss( out.view( out.size(0), -1), target ) 
print( out )
print( lsVal )

#lsVal.backward ()
#exit ()

g = autograd.grad( lsVal, [c1.parameters (), l1.parameters ()], create_graph=True )
gradient = torch.cat( [ grad.contiguous ().view( -1 ) for grad in g if grad is not None ] )

print( 'gradient is ', g )

vec = np.asarray( [ 1 for i in range( 1, len(gradient) + 1 ) ] )
vector = Variable( torch.from_numpy( vec ).type(torch.FloatTensor) )

print( 'Hessian Vector Being.... ')

gv = (vector * gradient).sum ()
hv = autograd.grad( gv, [ c1.parameters (), l1.parameters() ], create_graph=False )

print( 'Hessian Vector is : ', hv )
