

import numpy as np
import torch

print( 'Computing Batch Norm Derivative ')

err = np.asarray( [-0.6080235940 , 0.0599059907 , 0.4858675772 , 0.0622500264] )
zout = np.asarray( [0.8759333415, -1.0024917965 , 1.0906678043 , -0.9641093493 ] )

variance = 0.0003580263
epsilon = 1e-5

m = 4

print( 'Error in', err )
print( 'zout ', zout )

merr = m * err
print( 'm * err: ', merr )
serr = err.sum ()
print( 'sum_1: ', serr )

zerr = err.dot( zout ).sum ()

print( 'zerr: ', err * zout )
print( 'sum_2: ', err.dot( zout ).sum () )
zerr_z = zout * zerr

err_in = merr - serr 
print( 'merr - serr: ', err_in )
err_in = err_in - zerr_z; 
print(' - zerr_z : ', err_in )

err_in = err_in / (m * np.sqrt( variance + epsilon ))


print( err_in )

