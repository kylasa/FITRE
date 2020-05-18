
import numpy as np

epsilon = 1e-5
m = 4.

z_in = np.array( [0.82249, 0.82241, 0.81707, 0.85905] )
mu = np.array( [0.83026] )
var = np.array( [2.8114e-4] )

z_out = (z_in - mu )/pow( var + epsilon, .5)
print
print
print( 'z_out' )
print( z_out )

rz_in = np.array ( [1.2903, 1.3208, 1.3228, 1.5809] )
delta = np.array( [-.91106, .088538, .064736, .75778] )
pi = np.array( [.08893, .08853, .06473, .7577] )
#rdelta = np.array( [-.2305, -.0688, -.0914, .2079] )

delta_prime = m * delta - np.sum( delta, axis=0 ) - z_out * np.sum( delta * z_out, axis=0 )
delta_prime *= 1./(m * pow( var + epsilon, .5))
print
print
print( 'delta_prime')
print( delta_prime )

'''
rmu = (1./m) * np.sum( rz_in, axis=0 )
rop_first = (rz_in - rmu) * (1./np.sqrt( var + epsilon))

second = np.sum( (z_in - mu) * (rz_in - rmu), axis=0) * ((m-1.)/(m * m))
rop_second = ((z_in - mu) / pow( var + epsilon, 1.5)) * second

rz_out = rop_first - rop_second
'''

'''
 -0.2305  0.0914
 -0.0688  0.2079
'''

rmu = (1./m) * np.sum( rz_in, axis=0 )
rop_first = rz_in /np.sqrt( var + epsilon)

second = np.sum( (z_in - mu) * rz_in, axis=0) 
rop_second = ((z_in - mu) / (m * pow( var + epsilon, 1.5))) * second

rz_out = rop_first - rop_second

print
print
print( 'rz_out ' )
print( rz_out )

rdelta = pi * rz_out - pi * np.sum( pi * rz_out, axis=0 )

print
print
print( 'rdx' )
print( rdelta )

'''
[-0.23295289 -0.07092201  0.08919713  0.22361419]
'''


#This is the working formulation as of now... 
#rsecond = m * rdelta - np.sum( rdelta, axis=0 ) - z_out * np.sum( z_out * rdelta, axis=0 )



#rsecond = m * rdelta - np.sum( rdelta, axis=0 ) - rz_out * np.sum( delta * z_out, axis=0 ) - z_out * np.sum( z_out * rdelta + delta * rz_out, axis=0 )

#rsecond = m * rdelta - np.sum( rdelta, axis=0 ) - rz_out * np.sum( delta * z_out, axis=0 )# - z_out * np.sum( z_out * rdelta + delta * rz_out, axis=0 )

rsecond = m * rdelta - np.sum( rdelta, axis=0 ) - z_out * np.sum( z_out * rdelta, axis=0 )
rsecond = rsecond / (m * pow( var + epsilon, .5))

rfirst = m * delta - np.sum( delta, axis=0 ) - z_out * np.sum( delta * z_out, axis=0 )
rscale = np.sum( (z_in - mu) * rz_in , axis=0 ) * (1./( m *m)) * (-1./pow(var + epsilon, 1.5))

diff = (-z_out * np.sum( rz_out * delta, axis=0 ) - rz_out * np.sum( delta * z_out, axis=0 )) / (m * pow( var + epsilon, .5 ))

print
print
print
print ( delta )
print( rfirst )
print( rscale )
rfirst *= rscale
print( rfirst )


rdelta_prime = rsecond + rfirst


print
print
print( 'rdelta_prime' )
print( rdelta_prime )
print( rfirst )
print( rsecond )
print( diff )

'''
 339.8923 -148.6492
  -81.0532 -110.1900
'''

'''
 -10.7313  10.0752
  -1.2265   1.8826
'''
