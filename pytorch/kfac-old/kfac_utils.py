
import torch
import numpy as np


class KFACUtils: 

	def __init__( self, ksize, layers, batchSize, bias, debug, gamma ): 
		self.Zis = None
		self.dxA = None
		self.ZInv = []
		self.dInv = []
		self.layers = layers
		self.batchSize = batchSize

		self.kSize = ksize
		self.bias = bias
		self.debug = debug
		self.gamma = gamma

	def updateZAndD( self, z, d ): 
		self.Zis = z
		self.dxA = d

	def getPseudoInverse( self, mat ): 
		#Binv = np.linalg.pinv( mat )
		#return Binv
		return torch.inverse( mat )
		
	def computeInverses( self ): 

		# Zinvs here. 
		# d Inverses here.
		if (self.debug): 
			print( 'computeInverses: Begin' )
		self.ZInv = self.ZInv[:]
		self.dInv = self.dInv[:]
		for i in range( len( self.ZInv ) ): 
			mat = self.ZInv[ i ]
			#self.ZInv[ i ] = torch.from_numpy( self.getPseudoInverse( mat.cpu ().numpy ()  )).cuda () 
			self.ZInv[ i ] = self.getPseudoInverse( mat )
			mat = self.dInv[ i ]
			#self.dInv[ i ] = torch.from_numpy( self.getPseudoInverse( mat.cpu ().numpy ()  )).cuda ()
			self.dInv[ i ] = self.getPseudoInverse( mat )
		if (self.debug): 
			print( 'computeInverses: End' )

	def getExpandedForm( self, x ): 
		t = x.unfold( 2, self.kSize, 1 ).unfold( 3, self.kSize, 1)
		if(self.debug): 
			print( 'getExpandedForm: shaped--> ', t.size () )
		return t.permute( 0, 2, 3, 1, 4, 5 )

	def computeExpectation( self, x, height ): 
		if (self.debug): 
			print( 'computeExpectation: Mean denominator is: ', self.batchSize )
		return x / self.batchSize

	def matmulOmega( self, x ): 
		s = x.size ()
		if (self.debug): 
			print( 'matmulOmega: Computing Inputs to the layer ', s)
		fmtX = x.contiguous ().view( s[0] * s[1] * s[2], s[3] * s[4] * s[5] )
		d = fmtX.size ()
		if (self.bias == True): 
			fmtX = torch.cat( [fmtX, torch.ones( d[0], 1 ).type( torch.cuda.DoubleTensor )], dim=1 )

		fmtY = torch.mm( fmtX.permute(1, 0), fmtX )
		exp = self.computeExpectation( fmtY, s[3]*s[4]*s[5] )
		if (self.debug): 
			print( 'matmulOmega: Omega size  ', exp.size () )
		return exp

	def matmulLambda( self, l ): 
		s = l.size ()
		if (self.debug): 
			print( 'matmulLambda: computing derivatives  ', s )
		if (len(s) == 4): 
			fmtX = l.permute( 0, 2, 3, 1 )
			fmtX = fmtX.contiguous ().view( s[0] * s[2] * s[3], s[1] )
			fmtY = torch.mm( fmtX.permute( 1, 0 ), fmtX )
			exp = self.computeExpectation( fmtY, s[1] )
			if (self.debug):
				print( 'matmulLambda: derivativeSize  (Convolution)', exp.size () )
		else: 
			#perform l * l^T
			fmtY = torch.mm( l.permute( 1, 0 ), l )
			exp = self.computeExpectation( fmtY, s[1] )
			if (self.debug):
				print( 'matmulLambda: derivativeSize  (Linear)', exp.size () )
		return exp

	def matmul3d( self, x, linearize=False): 
		# in, samples
		s = x.size ()
		if (self.debug): 
			print( 'matmul3d: Linear Inputs... ', s )

		if (linearize == True): 
			fmtX = x.contiguous ().view( s[1] * s[2] * s[3], s[0] )

			if (self.bias == True): 
				ones = torch.ones( 1, s[ 0 ] ).type( torch.cuda.DoubleTensor )
				fmtX = torch.cat( [ fmtX, ones ], dim=0 )

			fmtY = torch.mm( fmtX, fmtX.permute( 1, 0 ))
			exp = self.computeExpectation( fmtY, s[0] )

			if (self.debug): 
				print( 'matmul3d: AA^T --> ', exp.size () )

		else: 

			if (self.bias == True): 
				ones = torch.ones( s[ 0 ], 1 ).type( torch.cuda.DoubleTensor )
				x = torch.cat( [ x, ones ], dim=1 )

			fmtY = torch.mm( x.permute( 1, 0), x )
			exp = self.computeExpectation( fmtY, s[0] )
			if (self.debug): 
				print( 'matmul3d: AA^T --> ', exp.size () )

		return exp


	def prepMatVec( self, X ): 
		c = []

		if (self.debug): 
			print( 'prepMatVec: Computing AA^T, GG^T, and inverses... ')

		for i in range( len( self.Zis ) ): 
			if (self.layers[i] == 'conv'): 
				if (i == 0):
					omega = self.matmulOmega( self.getExpandedForm( X ) )
				else: 
					omega = self.matmulOmega( self.getExpandedForm( self.Zis[ i ] ) )
			else: 
				if (self.layers[ i - 1] == 'conv'):
					omega = self.matmul3d( self.Zis[ i ], linearize=True )
				else:
					omega = self.matmul3d( self.Zis[ i ], linearize=False )

			cLambda = self.matmulLambda( self.dxA[ i ] )

			self.ZInv.append( omega + self.gamma * (torch.eye (omega.size ()[0]).type( torch.cuda.DoubleTensor ) ) )
			self.dInv.append( cLambda + self.gamma * (torch.eye (cLambda.size ()[0]).type( torch.cuda.DoubleTensor )) )

		#now compute the inverses
		if (self.debug): 
			print( 'prepMatVec: Done with storing intermediate results, starting Inverses' )
		self.computeInverses ()

	def computeMatVec( self, vec ): 

		#now compute the matrix-vec here.
		matvec = []

		if (self.debug): 
			print( 'computeMatVec: Starting Fisher Inverse X vector ')

		idx = 0
		for i in range( len(self.ZInv) ): 
			omega = self.ZInv[ i ]
			cLambda = self.dInv[ i ]

			vc = None
			if (self.layers[ i ] == 'conv' ): 
				s = vec[ idx ].size ()
				weights = vec[ idx ].view( s[0], s[1] * s[2] * s[3] )
				bias = torch.unsqueeze( vec[ idx + 1], dim=1)
				vc = torch.cat( [weights, bias], dim=1 )

				#lambda * vc
				temp = torch.mm( cLambda, vc )
				result = torch.mm( temp, omega )

				# Result is in shape outChannels X (inChannels * kSize * kSize)
				# Last column of this matrix is the bias vector
				resSize = result.size ()

				resWeight = result[:, 0:resSize[1]-1 ]
				resWeight = resWeight.contiguous ().view( s[0], s[1], s[2], s[3] )
				resBias = result[ :, -1 ]

				idx += 2
			else: 
				weights = vec[ idx ]
				biases = vec[ idx + 1]
				wSize = weights.size ()
				bSize = biases.size ()

				biases = torch.unsqueeze( biases, dim=1 )	
				vc = torch.cat( (weights, biases), dim=1 )
				temp = torch.mm( cLambda, vc )
				result = torch.mm( temp, omega )

				# Result is of the shape -- out * (in + 1 )
				# Last column is the bias vector... 

				resSize = result.size ()
				resWeight = result[ :, 0:resSize[1]-1 ]
				resWeight = resWeight.contiguous ().view( wSize )
				resBias = result[ :, -1 ]

				idx += 2

			matvec.append( resWeight)
			matvec.append( resBias )

		if (self.debug): 
			print( 'computeMatVec: Done... processing' )	

		return matvec
