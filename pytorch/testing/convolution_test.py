
import torch
import numpy as np

in_data = np.asarray( [ i for i in range( 1, 37 ) ] )
in_data = in_data.reshape( 6, 6 )
in_data = np.tile( in_data, (1, 3, 1, 1) )

conv = torch.nn.Conv2d( 3, 6, 3 )

w = np.asarray( [i for i in range( 1,  3 * 3 * 3 + 1 )]) * 0.01
w = np.tile( w, (6, 1) )
w = w.reshape( (6, 3, 3, 3) )

b = np.asarray( [ i for i in range( 1, 7) ] ) * 0.1


conv.weight.data.copy_ (torch.from_numpy( w ) )
conv.bias.data.copy_( torch.from_numpy( b ) )

out = conv( torch.autograd.Variable( torch.from_numpy( in_data ).type( torch.FloatTensor)) )
out = torch.nn.functional.softplus( out )
out = torch.nn.functional.avg_pool2d( out, 2 )

print (out )
