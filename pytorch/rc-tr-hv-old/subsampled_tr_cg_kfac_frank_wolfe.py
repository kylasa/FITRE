
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from frank_wolfe_update import getFrankWolfeUpdate


import numpy as np
import datetime

class SubsampledTRCGKFACFrank: 

	def __init__ (self, network, linsolver, params, stats, train, test, train_sampler ): 
		self.network = network
		self.train = train
		self.test = test
		self.params = params
		self.stats = stats
		self.linearSolver = linsolver
		self.train_sampler = train_sampler
		self.x = None

	def initializeZero( self ): 
		x = [ torch.zeros( W.numel() ).type( torch.DoubleTensor ) for W in self.network.parameters () ]
		self.x = torch.cat( x ).cuda ()

	def initializeRandom( self ): 
		#x = [ torch.rand( W.numel() ).type( torch.DoubleTensor ) for W in self.network.parameters () ]
		x = [ torch.randn( W.numel() ).type( torch.DoubleTensor ) for W in self.network.parameters () ]
		self.x = (0.25 * torch.cat( x )).cuda ()

	def initXavier( self ): 
		self.network.initXavierUniform()
		self.x = self.network.getWeights ()

	def initKaiming( self ): 
		self.network.initKaimingUniform()
		self.x = self.network.getWeights ()

	def getZeroVector (self): 
		z = [ torch.zeros( W.numel() ).type( torch.DoubleTensor ) for W in self.network.parameters () ]
		return torch.cat( [ p.contiguous().view(-1) for p in z ] ).cuda ()

	def getRandomVector (self): 
		r = [ torch.randn( W.numel() ).type( torch.DoubleTensor ) for W in self.network.parameters () ]
		#r = [ torch.ones( W.numel() ).type( torch.DoubleTensor ) for W in self.network.parameters () ]
		#r = [ torch.zeros( W.numel() ).type( torch.DoubleTensor ) for W in self.network.parameters () ]
		return ( torch.cat( [ p.contiguous().view(-1) for p in r ] )).cuda ()
		#return torch.cat( [ torch.ones( W.numel ()).type( torch.DoubleTensor ) for W in self.network.parameters () ] )


	def solve( self ): 

		self.network.setWeights( self.x.cuda () )

		i = 0
		failures = 0
		self.network.stopRecording ()

		# store the initial values of Omega and Lambda (KFAC) for 
		# large sample sizes:  5000
		large_sample_x, large_sample_y = next(iter( self.train_sampler ) )
		self.network.initKFACStorage( large_sample_x.shape[0] )
		self.network.startRecording ()
		self.network.computeTempsForKFAC ( large_sample_x.cuda () )
		self.network.computeInitialKFACTerms ( large_sample_x.cuda () )
		self.network.stopRecording ( )

		self.network.initKFACStorage( self.network.batchSize )

		train_set = self.moveToDevice( self.train )
		test_set = self.moveToDevice( self.test )

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

			X_sample_var = l[0]
			Y_sample_var = l[1]

			self.network.setWeights( self.x )

			self.network.zero_grad ()
			tr_ll, grad = self.network.computeGradientIter( X_sample_var, Y_sample_var  )
			grad = Variable( grad.type( torch.cuda.DoubleTensor) )

			#KFAC here. 
			#compute F{-1} * gradient = approximated natural gradient here. 
			#convert the vector to structures. 
			if (len(X_sample_var) == self.network.batchSize): 
				self.network.startRecording ()
				self.network.computeTempsForKFAC ( X_sample_var )
				self.network.stopRecording ()
				x_kfac_t = self.network.computeKFACHv( X_sample_var, self.network.unpackWeights( grad.data ), False)
			else: 
				x_kfac_t = self.network.computeKFACHv( X_sample_var, self.network.unpackWeights( grad.data ), True )

			x_kfac = torch.cat([ w.contiguous().view( -1 ) for w in x_kfac_t ])
			x_kfac = Variable( x_kfac.type( torch.cuda.DoubleTensor ) )

         # Frank-Wolfe Update
			frank_p, frank_m = getFrankWolfeUpdate( x_kfac, grad, self.network, self.params.delta, 20, 1e-8, X_sample_var, Y_sample_var )

			#for X, Y in self.train: 
			tr_ll, tr_accu = self.network.evalModel( X_sample_var, Y_sample_var )

			#import pdb;pdb.set_trace();
			self.network.updateWeights(  frank_p.data.mul_( kfac_step.numpy()[0] ) )

			#for X, Y in self.train: 
			self.network.zero_grad ()
			new_ll, new_accu = self.network.evalModel( X_sample_var, Y_sample_var )
			self.stats.no_props += len( self.train )
			rho = (tr_ll - new_ll) / (-frank_m.numpy ()[0])

			if (rho.data[0] > 0.75) : #1e-4 
				self.x = self.x.add( frank_p )
				self.params.delta = min( self.params.max_delta, self.params.gamma1 * self.params.delta ) # 2
			elif( rho.data[0] >= 0.25 ): 
				self.x = self.x.add( frank_p )
			else: 
				self.params.delta /= self.params.gamma1 # 2

			X_sample_var.volatile = True
			Y_sample_var.volatile = True
			x_kfac.volatile = True
			step.volatile = True
			m_g_kfac.volatile = True
			m_kfac.volatile = True
			grad.volatile = True
