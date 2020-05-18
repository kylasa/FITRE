
import torch
import numpy as np

#temp(x).unfold( 2, self.kSize, 1 ).unfold( 3, self.kSize, 1)

'''
w = np.asarray( [i for i in range( 1,  3 * 3 * 3 + 1 )]) * 0.01
w = np.tile( w, (6, 1) )
w = w.reshape( (6, 3, 3, 3) )
'''

mat = np.asarray( [i for i in range( 1, 6*6 + 1) ] )
mat = np.tile( mat, (10, 3, 1) )
mat = mat.reshape( (10, 3, 6, 6) )

for i in range( 10): 
	mat[i, 1,:,: ] *= 10
	mat[i, 2,:,: ] *= 100

print( 'Input matrix is-->', mat )

mat = torch.from_numpy( mat )

exp = mat.unfold( 2, 3, 1).unfold( 3, 3, 1)
exp = exp.permute( 0, 2, 3, 1, 4, 5)
s = exp.size ()
exp = exp.contiguous ().view( s[0]* s[1] * s[2], s[3] * s[4] * s[5] )

print( 'expanded matrix is', exp )

print( 'expanded matrix first ', exp[ 0, : ] )
print( 'expanded matrix first ', exp[ 1, : ] )

aat = exp.t() @ exp

print( 'Result is --> ', aat )
