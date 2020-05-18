
import torch
import numpy as np


class KFACUtils: 

	def __init__( self, ksize, layers, batchSize, bias, debug, gamma, theta, regLambda ): 
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
		self.theta = theta
		self.regLambda = regLambda

		self.thetaZ = None
		self.thetaD = None

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
		for i in range( len( self.ZInv ) ): 
			mat = self.thetaZ[ i ]
			self.thetaZ[ i ] = self.getPseudoInverse( mat )

			mat = self.thetaD[ i ]
			self.thetaD[ i ] = self.getPseudoInverse( mat )

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
			num_spatial_locations = s[2] * s[3]
			fmtX = l.permute( 0, 2, 3, 1 )
			fmtX = fmtX.contiguous ().view( s[0] * s[2] * s[3], s[1] )
			fmtY = torch.mm( fmtX.permute( 1, 0 ), fmtX )
			exp = self.computeExpectation( fmtY, s[1] * num_spatial_locations )
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


	def prepMatVec( self, X, ignoreInverses=False): 
		c = []
		
		if (self.debug): 
			print( 'prepMatVec: Computing AA^T, GG^T, and inverses... ')

		self.ZInv = []
		self.dInv = []
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

			# Add the gamma term here. 
			# This should be before the inverse... 
			# compute pi_l and use the scaling term appropriately
			# Omega = pi_l * sqrt( gamma + lambda )
			# Lambda = 1/pi_l *sqrt( gamma + lambda) 
			# pi_l = 1 -- Simple approach, traceNorm(.) etc...
			scale = torch.sqrt( torch.from_numpy( np.asarray( [ self.regLambda + self.gamma ] ) ).type( torch.cuda.DoubleTensor ) )
			omega +=  scale * torch.eye( omega.size()[0]).type( torch.cuda.DoubleTensor )
			cLambda += scale * torch.eye( cLambda.size()[0]).type( torch.cuda.DoubleTensor )

			self.ZInv.append( omega )
			self.dInv.append( cLambda )

		# Store the initial results in the very first iteration 
		# This is for large batch size -- discussed in the paper
		# to Initialize the Zis and Dis 
		# compute the Moving average here. 
		# Theta is the moving average. 
		if self.thetaZ is not None: 
			for idx in range( len( self.thetaZ) ): 
				self.thetaZ[ idx ] = self.theta * self.thetaZ[ idx ] + (1. - self.theta) * self.ZInv[ idx ]
				self.thetaD[ idx ] = self.theta * self.thetaD[ idx ] + ( 1. - self.theta ) * self.dInv[ idx ]
		else: 
			self.thetaZ = self.ZInv
			self.thetaD = self.dInv

		#now compute the inverses
		if (self.debug): 
			print( 'prepMatVec: Done with storing intermediate results, starting Inverses' )

		if (not ignoreInverses): 
			self.computeInverses ()

	def computeMatVec( self, vec ): 

		#now compute the matrix-vec here.
		matvec = []

		if (self.debug): 
			print( 'computeMatVec: Starting Fisher Inverse X vector ')

		idx = 0
		for i in range( len(self.thetaZ) ): 
			omega = self.thetaZ[ i ]
			cLambda = self.thetaD[ i ]

			vc = None
			if (self.layers[ i ] == 'conv' ): 
				s = vec[ idx ].size ()
				weights = vec[ idx ].view( s[0], s[1] * s[2] * s[3] )
				if (self.bias): 
					bias = torch.unsqueeze( vec[ idx + 1], dim=1)
					vc = torch.cat( [weights, bias], dim=1 )
				else: 
					vc = weights

				#lambda * vc
				temp = torch.mm( cLambda, vc )
				result = torch.mm( temp, omega )

				# Result is in shape outChannels X (inChannels * kSize * kSize)
				# Last column of this matrix is the bias vector
				if (self.bias): 
					resSize = result.size ()
					resWeight = result[:, 0:resSize[1]-1 ]
					resWeight = resWeight.contiguous ().view( s[0], s[1], s[2], s[3] )
					resBias = result[ :, -1 ]
					idx += 2
				else: 
					resWeight = result
					idx += 1

			else: 
				weights = vec[ idx ]
				wSize = weights.size ()

				if( self.bias ): 
					biases = vec[ idx + 1]
					bSize = biases.size ()
					biases = torch.unsqueeze( biases, dim=1 )	
					vc = torch.cat( (weights, biases), dim=1 )
				else: 
					vc = weights

				#print( cLambda.shape, vc.shape, omega.shape )
				temp = torch.mm( cLambda, vc )
				result = torch.mm( temp, omega )

				# Result is of the shape -- out * (in + 1 )
				# Last column is the bias vector... 

				if (self.bias): 
					resSize = result.size ()
					resWeight = result[ :, 0:resSize[1]-1 ]
					resWeight = resWeight.contiguous ().view( wSize )
					resBias = result[ :, -1 ]
					idx += 2
				else: 
					resWeight = result
					idx += 1

			matvec.append( resWeight)
			if (self.bias): 
				matvec.append( resBias )

		if (self.debug): 
			print( 'computeMatVec: Done... processing' )	

		return matvec
