
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from utils import group_product
from utils import group_add


import numpy as np
import datetime
import math

from frank_wolfe_update import getFrankWolfeUpdate

class SubsampledTRCGKFACFrankWolfe: 

	def __init__ (self, network, linsolver, params, stats, train, test, train_sampler, momentum, check_grad, debug ): 
		self.network = network
		self.train = train
		self.test = test
		self.params = params
		self.stats = stats
		self.linearSolver = linsolver
		self.train_sampler = train_sampler
		self.momentum = momentum
		self.check_grad = check_grad
		self.debug = debug

	def initializeZero( self ): 
		for p in self.network.parameters ():
			p.data.fill_( 0 )

	def initializeRandom( self ): 
		x = [ torch.randn( W.numel() ).type( torch.DoubleTensor ) for W in self.network.parameters () ]

	def initXavier( self ): 
		self.network.initXavierUniform()

	def initKaiming( self ): 
		self.network.initKaimingUniform()
	
	def initConstant( self ): 
		self.network.initconstant ()

	def getZeroVector (self): 
		z = [ torch.zeros( W.numel() ).type( torch.DoubleTensor ) for W in self.network.parameters () ]
		return torch.cat( [ p.contiguous().view(-1) for p in z ] ).cuda ()

	def getRandomVector (self): 
		r = [ torch.randn( W.numel() ).type( torch.DoubleTensor ) for W in self.network.parameters () ]
		return ( torch.cat( [ p.contiguous().view(-1) for p in r ] )).cuda ()


	def solve( self ): 

		i = 0
		failures = 0

		train_set = self.moveToDevice( self.train )
		test_set = self.moveToDevice( self.test )
		self.network.initKFACStorage( )

		self.prevWeights = []
		for p in self.network.parameters (): 
			self.prevWeights.append( torch.zeros_like( p ) )

		while (i <= self.params.max_iters): 

			start = datetime.datetime.now ()

			# Make one full pass of the dataset here. 
			self.fullPass (train_set)

			end = datetime.datetime.now ()

			# Compute Train and Test Accuracy here. 
			train_ll, tr_accu = self.getStatistics (train_set)
			test_ll, test_accu = self.getStatistics (test_set)
					
			#print stats here.
			self.stats.train_accu = tr_accu
			self.stats.test_accu = test_accu

			self.stats.train_ll = train_ll
			self.stats.test_ll = test_ll

			self.stats.delta = 0
			self.stats.gradnorm = 0

			self.stats.tr_failures = 0 
			self.stats.cg_iterations = 0 

			self.stats.iteration_time = ( end - start ).total_seconds ()
			self.stats.total_time += (end - start ).total_seconds ()

			self.stats.printIterationStats ()

			i += 1

		#cleanup and exit....
		self.stats.shutdown ()

	def getStatistics (self, dataset): 

		ll = 0
		accu = 0
		for l in dataset:
			t, a = self.network.evalModel( l[0], l[1])
			ll += t
			accu += a
		return ll / len( dataset ), accu / len( dataset )

	def moveToDevice (self, dataset ): 
		l = []
		for x, y in dataset: 
			l.append( (x.type(torch.DoubleTensor).cuda(), y.cuda()) )
		return l


	def fullPass( self, train ): 

		for l in train: 

			torch.cuda.empty_cache ()
			print( 'Starting the full pass... ')

			X_sample_var = l[0]
			Y_sample_var = l[1]
			tr_ll, tr_accu = self.network.evalModel( X_sample_var, Y_sample_var )

			# With regularization
			self.network.zero_grad ()
			grad = self.network.computeGradientIter2( X_sample_var, Y_sample_var  )

			#KFAC here. 
			self.network.zero_grad ()
			self.network.startRecording ()
			self.network.computeTempsForKFAC ( X_sample_var )
			self.network.stopRecording ()
			x_kfac = self.network.computeKFACHv( X_sample_var, grad )

			# assert here, KFAC is PSD
			if (group_product( grad, x_kfac ) < 0 ): 
				print( 'Trouble with KFAC direction... it is NOT PSD ' )
				exit ()

			#Use Frank-Wolfe Direction here. 
			fw_direction = []
			for t in grad: 
				fw_direction.append( torch.zeros( t.size () ).type( torch.cuda.DoubleTensor ) )

			fw_direction, fw_model = getFrankWolfeUpdate( fw_direction, x_kfac, self.network, self.params.delta, 20, 1e-8, X_sample_var, Y_sample_var )

			for idx, p in enumerate( self.prevWeights ): 
				fw_direction[idx].add_ (self.momentum, p )
					
			for idx, w in enumerate( self.network.parameters () ): 
				w.data.add_( -1., fw_direction [ idx ] )
				self.prevWeights[ idx ].copy_( fw_direction[ idx ] )

			new_ll, new_accu = self.network.evalModel( X_sample_var, Y_sample_var )
			rho = (new_ll - tr_ll) / ( fw_model - 1e-16 )

			if (rho > 0.75) : #1e-4 
				self.params.delta = min( self.params.max_delta, 2. * self.params.delta ) # 2
			if (rho < 0.25): 
				self.params.delta = max( self.params.min_delta, 0.5 * self.params.delta ) # 2
			if ((rho < 1e-4) or (new_ll > (10. * tr_ll))): 
				if(self.debug): 
					print( 'Trouble.... Reject this step, since there  is no VISIBLE decrease ' )

			for idx, w in enumerate( self.network.parameters () ):
				w.data.add_ (1., self.prevWeights[ idx ] )
			if (self.debug): 
				print( rho, self.params.delta, new_ll, tr_ll, fw_model)

			del grad
