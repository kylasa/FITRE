
import numpy as np

def get4DMatrix( mat, size ): 
	#print mat, size
	outChannels = size[ 0 ]
	inChannels = size[ 1 ]
	height = size[ 2 ]
	width = size[ 3 ]

	result = np.zeros( (outChannels, inChannels, height, width) )

	offset = 0
	for o in range( outChannels ): 
		for i in range( inChannels ): 
			#print( o, i, height, width, offset, offset + height * width )
			chunk = mat[ offset:offset + height*width ]
			#print( chunk )
			p = chunk.reshape( height, width )
			result[ o, i, :, : ] = p.T

			offset += height * width

			#print o, i, p.T
	return result

def get2DMatrix( mat, size ): 
	#print mat, size
	height = size[ 0 ]
	width = size[ 1 ]

	result = np.empty( (height, width) )
	result = mat[ 0:height*width ]
	result = result.reshape( (height, width), order='F' )
	
		
	return result

def get1DMatrix( mat, size ): 
	#print mat, size
	return mat[ 0:size[0] ]

def readMatrix( fName, sizes ): 
	
	a = np.loadtxt( fName, dtype=np.float64 )
	out = []

	length = len( a )
	print (length)
	offsets = [0]
	for i in sizes: 
		offsets.append( np.prod( i ) )

	offsets = np.cumsum( offsets )

	print (offsets)

	index = 0
	for s in sizes: 
		print (s)
		if len(s) == 4: 
			out.append( get4DMatrix( a[offsets[ index ]: offsets[ index+1 ] ], s ))
		if len(s) == 2: 
			out.append( get2DMatrix( a[offsets[ index ]: offsets[ index+1 ] ], s ))
		if len(s) == 1: 
			out.append( get1DMatrix( a[offsets[ index ]: offsets[ index+1 ] ], s ))

		index += 1

	return out

#
#	test the code here. 
#
'''
f = np.asarray( [ i for i in range( 1, 43 ) ] )

np.savetxt( 'sample.txt', f )

a = np.loadtxt( 'sample.txt' )

result = readMatrix( 'sample.txt', [ [2, 2, 2, 2], [2], [2, 2, 2, 2], [2], [2, 2], [2] ] )

print result
'''
