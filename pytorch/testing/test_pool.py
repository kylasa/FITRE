import torch
import numpy as np
import math
from torch import nn
from torch.autograd import Variable

def POOLHOOK( module, grad_input, grad_output ):
   print( 'grad input', grad_input )
   print( 'grad out', grad_output )

h = 4
w = 4
stride = 2
pad = 1
kernel = 3

hout = math.floor( (h + 2 * pad - kernel)/stride + 1 )
wout = math.floor( (w + 2 * pad - kernel)/stride + 1 )


pool_layer = nn.AvgPool2d( kernel_size = kernel, stride = stride, padding = pad )
pool_layer.register_backward_hook( POOLHOOK )

in_data = torch.ones( 4, 4 )
in_data = torch.unsqueeze( in_data, dim=0 )
print( 'Input data.... ' )
print( in_data )
print()
print()

out_data = pool_layer( Variable( in_data, requires_grad=True ) )

print( 'Output data... ' )
print( out_data )

print( 'Gradient computation... ' )
print ()

target = torch.ones( out_data.size () )
#print ('Loss: ', (out_data - Variable( target ) ) )

loss = torch.nn.MSELoss( )
#lossVal = loss( out_data, Variable( target ) )
lossVal = (out_data - Variable( target.view( out_data.size () ), requires_grad=True )).sum ()
lossVal.backward ()

print ()
print ()
print ()
print ('Done... ' )
