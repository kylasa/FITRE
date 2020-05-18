import math
import torch
import numpy as np
from torch import nn
from utils import group_product

TYPE = torch.cuda.FloatTensor


class KFACUtils: 

	def __init__( self, ksize, padding, layers, batchSize, bias, debug, gamma, regLambda, stats_decay ): 
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
		self.regLambda = regLambda
		self.padding = padding
		self.stats_decay = stats_decay

		self.thetaZ = None
		self.thetaD = None

	def updateZAndD( self, z, d ): 
		self.Zis = z
		self.dxA = d

	def getPseudoInverse( self, mat ): 
		#Binv = np.linalg.pinv( mat )
		#return Binv
		#return torch.inverse( mat )
		X, LU = torch.gesv( torch.eye(mat.size(0)).type( TYPE ) , mat )
		return X
		

	def computeInverses( self ): 

		# Zinvs here. 
		# d Inverses here.
		if (self.debug): 
			print( 'computeInverses: Begin' )

		#scale = torch.from_numpy( np.asarray( [ self.regLambda + self.gamma ] ) ).type( torch.cuda.DoubleTensor )
		scale = self.regLambda + self.gamma
		for i in range( len( self.thetaZ) ): 
			mat = self.thetaZ[ i ] + scale * torch.eye( self.thetaZ[i].size ()[0] ).type( TYPE )
			#print( mat[0:10][0] )
			#print( 'Z[i]: ', math.sqrt( group_product( mat, mat ) ) )
			#print( 'Sigma Z[i]: ', torch.sum( torch.sum( mat, dim=0 ), dim=0) )
			self.ZInv.append( self.getPseudoInverse( mat ) )
			#print( 'ZInv[i]: ', math.sqrt( group_product( self.ZInv[i], self.ZInv[i] ) ) )
			#print( 'ZZT-->', mat )
			#print( 'ZInv-->', self.ZInv[ i ] )

			mat = self.thetaD[ i ] + scale * torch.eye( self.thetaD[i].size ()[0] ).type( TYPE )
			#print( mat[0:10][0] )
			#print( 'D[i]: ', math.sqrt( group_product( mat, mat ) ) )

			self.dInv.append( self.getPseudoInverse( mat ) )
			#print( 'DInv[i]: ', math.sqrt( group_product( self.dInv[i], self.dInv[i] ) ) )
			#print( 'D--> ', mat )
			#print( 'dInv-->', self.dInv[ i ] )

		if (self.debug): 
			print( 'computeInverses: End' )

	def getExpandedForm( self, x ): 
		
		if (self.padding != 0):
			temp=nn.ZeroPad2d(self.padding)
			t = temp(x).unfold( 2, self.kSize, 1 ).unfold( 3, self.kSize, 1)
		else: 
			t = x.unfold( 2, self.kSize, 1 ).unfold( 3, self.kSize, 1)

		if(self.debug): 
			print( 'getExpandedForm: shaped--> ', t.size () )

		ret = t.permute( 0, 2, 3, 1, 4, 5 )

		#print( 'Norm of expaneded Z', torch.norm( ret ) )

		return ret

	def update_running_stats( self, aa, m_aa, momentum): 
		if (momentum == 0): 
			m_aa.copy_( aa )
		else: 
			#m_aa *= momentum / (1 - momentum )
			m_aa = m_aa.mul_( momentum/ (1- momentum) )
			m_aa += aa
			#m_aa *= (1 - momentum )
			m_aa = m_aa.mul_( 1 - momentum )

		return m_aa


	def matmulOmega( self, x, m_aa ): 
		s = x.size ()
		if (self.debug): 
			print( 'matmulOmega: Computing Inputs to the layer ', s)
		fmtX = x.contiguous ().view( s[0] * s[1] * s[2], s[3] * s[4] * s[5] )
		#print ( fmtX[ 0 ][ : ] )
		#fmtX = fmtX / (s[1] * s[2] )
		fmtX = fmtX.div_( s[1] ).div_( s[2] )
		#print ( fmtX[ 0 ][ : ] )
		d = fmtX.size ()
		if (self.bias == True): 
			fmtX = torch.cat( [fmtX, torch.ones( d[0], 1 ).type( TYPE )], dim=1 )

		#fmtY = torch.mm( fmtX.permute(1, 0), fmtX )
		#exp = self.computeExpectation( fmtY, s[3]*s[4]*s[5] )

		# final updated value
		#fmtY = fmtY.div_( s[0] )
		fmtY = fmtX.t() @ ( fmtX / s[0] )

		if (self.debug): 
			print( 'matmulOmega: Omega size  ', fmtY.size () )
			#print( ' First 10 elements: ', fmtY[:, 0][0:9] )

		#print( 'ZZT matrix is --> ', fmtY )

		# Now take care of the stats_decay term here. 
		if m_aa is None: 
			#m_aa = torch.zeros( fmtY.shape ).type( torch.cuda.DoubleTensor )
			m_aa = fmtY.clone ()
		m_aa = self.update_running_stats( fmtY, m_aa, self.stats_decay )

		return m_aa

	def matmulLambda( self, fmtX, m_aa ): 
		# 200, 64, 32, 32
		# 200, 64, 16, 16
		#print( fmtX.size () )
		#print( 'Lambda Raw: ', math.sqrt( group_product( fmtX, fmtX ) ) )
		s = fmtX.size ()
		#print( s )
		if (self.debug): 
			print( 'matmulLambda: computing derivatives  ', s )
		if (len(s) == 4): 
			#num_spatial_locations = s[2] * s[3]
			#print( num_spatial_locations )
			#fmtX = l.permute( 0, 2, 3, 1 )
			#fmtX = fmtX.contiguous ().view( s[0] * s[2] * s[3], s[1] ).mul_( s[2] ).mul_( s[3] ) 
			#fmtX = fmtX.contiguous ().view( -1, fmtX.size(-1)).mul_( fmtX.size(1) ).mul_( fmtX.size(2) ) 
			#print( "Orignal Dxs stored from probs distribution... \n", fmtX )
			fmtX = fmtX.transpose(1,2).transpose(2,3).contiguous ()
			#print( fmtX.size () )
			fmtX = fmtX.view( -1, fmtX.size(-1) ).mul_( fmtX.size(1) ).mul_( fmtX.size(2) )
			#fmtY = torch.mm( fmtX.permute( 1, 0 ), fmtX )
			_fmtX = fmtX * s[0]
			#print( s, fmtX.size() )
			#print( _fmtX )
			exp = _fmtX.t () @ (_fmtX / fmtX.size(0) )

			#exp = fmtY.mul_( s[2] ).mul_( s[3] )
			#exp = self.computeExpectation( fmtY, s[1] * num_spatial_locations )
			if (self.debug):
				print( 'matmulLambda: derivativeSize  (Convolution)', exp.size () )
		else: 
			#print( fmtX )
			#perform l * l^T
			#exp = torch.mm( l.permute( 1, 0 ), l )
			exp = fmtX.t() @ (fmtX * s[0] )
			#exp = self.computeExpectation( fmtY, s[1] )
			if (self.debug):
				print( 'matmulLambda: derivativeSize  (Linear)', exp.size () )

		#print( 'before Momentum: ', math.sqrt( group_product( exp, exp ) ) )

		# Now take care of the stats_decay term here. 
		if (m_aa is None): 
			#m_aa = torch.zeros( exp.shape ).type( torch.cuda.DoubleTensor )
			m_aa = exp.clone ()
		m_aa = self.update_running_stats( exp, m_aa, self.stats_decay )

		return m_aa

	def matmul3d( self, x, linearize, m_aa): 
		# in, samples
		s = x.size ()
		if (self.debug): 
			print( 'matmul3d: Linear Inputs... ', s )

		if (linearize == True): 
			#fmtX = x.contiguous ().view( s[1] * s[2] * s[3], s[0] )
			fmtX = x.contiguous ().view( s[0], -1 )

			if (self.bias == True): 
				ones = torch.ones( s[0], 1 ).type( TYPE )
				fmtX = torch.cat( [ fmtX, ones ], dim=1 )

			#exp = torch.mm( fmtX, fmtX.permute( 1, 0 )) / s[0]
			#print( fmtX[0:1][0])
			#print( fmtX.size () )
			#print( 'Norm( Z, 2) == ', torch.norm( fmtX ) )
			exp = fmtX.t() @ (fmtX / s[0] )
			#print( 'Norm( ZZT, 2) == ', torch.norm( exp ) )
			#exp = self.computeExpectation( fmtY, s[0] )
			#print( exp )

			if (self.debug): 
				print( 'matmul3d: AA^T --> ', exp.size () )

		else: 

			if (self.bias == True): 
				ones = torch.ones( s[ 0 ], 1 ).type( TYPE )
				x = torch.cat( [ x, ones ], dim=1 )

			#print( 'Norm( Z, 2) == ', torch.norm( x ) )

			#exp = torch.mm( x.permute( 1, 0), x ) / s[0]
			exp = x.t() @ (x / s[0] )
			#print( 'Norm( ZZT, 2) == ', torch.norm( exp ) )
			#exp = self.computeExpectation( fmtY, s[0] )

			#print( exp )

			if (self.debug): 
				print( 'matmul3d: AA^T --> ', exp.size () )

		if (m_aa is None): 
			#m_aa = dtorch.zeros( exp.shape ).type( torch.cuda.DoubleTensor )
			m_aa = exp.clone ()
		m_aa = self.update_running_stats( exp, m_aa, self.stats_decay )

		return m_aa


	def prepMatVec( self, X ): 
		c = []
		
		if (self.debug): 
			print( 'prepMatVec: Computing AA^T, GG^T, and inverses... ')

		if self.thetaZ is None: 
			self.thetaZ = []
			for x in self.Zis: 
				self.thetaZ.append( None )

			self.thetaD = []
			for x in self.Zis: 
				self.thetaD.append( None )

		for i in range( len( self.Zis ) ): 
			if (self.layers[i] == 'conv'): 
				if (i == 0):
					self.thetaZ[i] = self.matmulOmega( self.getExpandedForm( X ), self.thetaZ[ i ] )
				else: 
					self.thetaZ[i] = self.matmulOmega( self.getExpandedForm( self.Zis[ i ] ), self.thetaZ[ i ] )
			else: 
				if (self.layers[ i - 1] == 'conv'):
					self.thetaZ[i] = self.matmul3d( self.Zis[ i ], True, self.thetaZ[ i ] )
				else:
					self.thetaZ[i] = self.matmul3d( self.Zis[ i ], False, self.thetaZ[ i ] )

			self.thetaD[i] = self.matmulLambda( self.dxA[ i ], self.thetaD[ i ] )

		#now compute the inverses
		if (self.debug): 
			print( 'prepMatVec: Done with storing intermediate results, starting Inverses' )

		del self.ZInv
		del self.dInv
		self.ZInv = []
		self.dInv = []
		self.computeInverses ()
		torch.cuda.empty_cache ()

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
				if (self.bias): 
					bias = torch.unsqueeze( vec[ idx + 1], dim=1)
					vc = torch.cat( [weights, bias], dim=1 )
				else: 
					vc = weights

				#lambda * vc
				#temp = torch.mm( cLambda, vc )
				#result = torch.mm( temp, omega )
				#print( 'vector component: ', torch.norm( vc )) 
				result = cLambda @ vc @ omega

				#print( 'Natural Gradient: ', vc )
				#print( 'Original Gradient: ', vec[ idx ] )

				#result = result.view( s )
				#print( math.sqrt( group_product( result, result ) ) )
				#print( torch.norm( result )) 

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
					resWeight = resWeight.contiguous ().view( s[0], s[1], s[2], s[3] )
					idx += 1

				#print( 'Vector Convolution: ', resWeight )
				#print( resBias )


				#print( vc.shape, result.shape )
				#print( torch.dot( vc.view(-1), result.view(-1) ) )

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

				#print( 'Linear Vector: ', vc )

				#print( cLambda.shape, vc.shape, omega.shape )
				#temp = torch.mm( cLambda, vc )
				#result = torch.mm( temp, omega )


				#if i == 4: 
					#result = result.view( wSize )
					#print( 'This is cLambda: ', cLambda )
					#print( 'This is vector: ', vc )
					#print( 'This is result: ', result )
					#print( 'This is omega: ', omega )

			
				result = cLambda @ vc @ omega
				#print( torch.norm( result )) 
				'''
				if i == 4: 
					#print( 'Update resulte is :', result )
					print( 'vc: ', vc )
					print( 'cLambda: ', cLambda )
					print( 'Omega: ', omega )
					print( 'Result: ', result )
					print( 'cLambda @ vc : ', cLambda @ vc )
					print( 'vc @ omega : ', vc @ omega )
					print( cLambda.size () )
					print( vc.size () )
					print( omega.size () )
					print( result.size () )
					print( math.sqrt( group_product( cLambda, cLambda) ), math.sqrt( group_product( vc, vc ) ), math.sqrt( group_product( omega, omega ) ) )
				'''


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
					resWeight = resWeight.contiguous ().view( wSize )
					idx += 1
				#print( torch.dot( vc.view(-1), result.view(-1) ) )

				#print( 'Natural Gradient Linear: ', resWeight )
				#print(  resBias )

			matvec.append( resWeight)
			if (self.bias): 
				matvec.append( resBias )


		if (self.debug): 
			print( 'computeMatVec: Done... processing' )	

		return matvec
