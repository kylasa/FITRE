
import numpy as np
import torch

print( 'Computing Batch Norm Derivative ')

m = 4

means = 0.8285777243 
variance = 0.0004594150 
epsilon = 1e-5

zin = np.asarray ( [ 0.8644751360 , 0.8250697742 , 0.8099140393 , 0.8148519478 ] )
zout = np.asarray( [ 1.6568555953 , -0.1619104702 , -0.8614278725 , -0.6335172527 ] )

rzin = np.asarray( [ 1.5819700810 , 1.0040864902 , 0.9148101685, 1.0778613252 ] )
rzout = np.asarray( [ 1.3013500693, -4.6440685046 , -0.7928236666 , 4.1355421018 ] )

err = np.asarray( [ -0.2559813623 , 0.1206990214 , 0.0599662929 , 0.0753160484 ] )
rerr = np.asarray( [ 0.4685278021 , -0.6415989045 , -0.0878175082 , 0.2608886103 ] )


rop_means = rzin.sum () / m


merr = m * err
serr = err.sum ()

zerr = err.dot( zout ).sum ()
zerr_z = zout * zerr

err_in = merr - serr 
err_in = err_in - zerr_z; 

t1 = zin - means
t2 = rzin - rop_means

print( 't1 * t2: ', (t1 * t2).sum () )
print( 'Denominator: ', pow( variance + epsilon, 1.5) )

r1 = -(1./(m * m)) * (1./(pow( variance + epsilon, 1.5))) * (t2 * t1).sum ()

scaledR1 = r1 * err_in

print( 'First Term: ', scaledR1 )

#
# second term here. 
#

r2 = m * rerr
r2 = r2 - rerr.sum ()
r2 = r2 - rzout * ( err.dot( zout ).sum () )
r2 = r2 - zout * (rerr.dot( zout) + rzout.dot( err)).sum ()

print( 'Derivative of numerator: ', r2 )

r2 = r2 * (1./(m * np.sqrt( variance + epsilon)))

result = r1 + r2
print( result )
