
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

conv = nn.Conv2d(3, 6, 5, bias=False)

size = 6
#X = torch.rand(1, 3, size, size ) 
X = torch.from_numpy( np.asarray( [ (i /(32 * 32) ) for i in range(32 * 32 * 3) ] ).reshape( (1, 3, 32, 32)) ).type( torch.FloatTensor )
out = conv( Variable(X) )

print( X.size () )

#Y = F.unfold( X, kernel_size=3 )
Y = X.unfold(2, 5, 1).unfold(3, 5, 1)
Y = Y.permute( 0, 2, 3, 1, 4, 5 )
print( Y.size () )
print
print
print
print


Y = Y.squeeze (0).squeeze(0)
print( Y.size() )

s = Y.size ()

Z = Y.contiguous ().view( s[0] * s[1], s[2] * s[3] * s[4] )
print( Z.size () )

R = Z.view( s[0], s[1], s[2], s[3], s[4] )
print( R.size () )



'''
Z = torch.mm( Y.permute(2, 3, 4, 0, 1), Y )
print( Z.size() )
'''
