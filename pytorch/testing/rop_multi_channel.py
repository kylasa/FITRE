
import numpy as np

epsilon = 1e-5
m = 4.

z_in = np.array( [
[ 1.0370973948,  1.0043139587,  0.9827999882,  1.1734451854 ], 
[ 1.0263511413,  0.9444544079,  0.9624067324,  1.1297179393 ], 
[ 0.9839044594,  0.9824748661,  0.9652912248,  1.1107669652 ], 
[ 1.0940783854,  1.0353194030,  1.0174183545,  1.2496661204 ] ] )

rz_in = np.array ( [
[ 3.4514463613,  3.4508039526,  3.6962432230,  4.4474638143 ], 
[ 3.1004567772,  2.5445607883,  3.5196310807,  4.1068548876 ], 
[ 2.9800273099,  3.2689161083,  3.6822395871,  3.6194282985 ], 
[ 4.0192577776,  4.2434160529,  4.3751819995,  4.6792050747 ]
])

means = np.array( [ 
1.0353578452 , 0.9916406589 , 0.9819790750 , 1.1658990526 
] )

variances = np.array( [ 
0.0015449249 , 0.0010947498 , 0.0004795443 , 0.0028556214 
] )


rmu = (1./m) * np.sum( rz_in, axis=0 )
print ()
print( 'RMU: ', rmu )

rop_first = (rz_in - rmu) /np.sqrt( variances + epsilon)
print ()
print( 'ROP_FIRST: ', rop_first )

second = np.sum( (z_in - means) * (rz_in - rmu), axis = 0 )
print ()
print( 'Second: ', second )
rop_second = ((z_in - means) / (m * pow( variances + epsilon, 1.5))) * second

print ()
print( 'ROP_SECOND: ', rop_second )

rz_out = rop_first - rop_second

print ()
print ()
print( 'rz_out ' )
print( rz_out )
