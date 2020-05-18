
import os

damp = [ 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1  ]
tr = [ 100 ]
regularization = [ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2 ]

with open( './imagenet_test.sh', 'w' ) as f: 
	f.write( '#!/bin/bash\n' )

	for d in damp: 
		for r in tr: 	
			for reg in regularization: 
				f.write( './NewtonTRSolver 8 200 ' + str( d ) + ' ' + str( r ) + ' 0 0 0 0 1 5 3 10 3 '+ str( reg ) + '&> image_'+ str( d ) + '_' + str( r ) +  '_' + str( reg) +'.log\n\n' )


