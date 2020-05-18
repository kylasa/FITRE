
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable


import numpy as np
import datetime

class SubsampledTRCGKFAC: 

	def __init__ (self, network, linsolver, params, stats, train, test, train_sampler, initialization, scale ): 
		self.network = network
		self.train = train
		self.test = test
		self.params = params
		self.stats = stats
		self.linearSolver = linsolver
		self.train_sampler = train_sampler
		self.x = None
		self.initialization = initialization
		self.scale = scale

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

		delta = self.params.delta

		self.network.initZeroWeights( )
		self.network.updateWeights( self.x.cuda () )
		#for i in range( self.params.max_iters ): 
		i = 0
		failures = 0
		self.network.stopRecording ()

		# store the initial values of Omega and Lambda (KFAC) for 
		# large sample sizes:  5000
		s1X, s1Y = next(iter( self.train_sampler ) )
		s2X, s2Y = next(iter( self.train_sampler ) )
		s3X, s3Y = next(iter( self.train_sampler ) )
		s4X, s4Y = next(iter( self.train_sampler ) )
		s5X, s5Y = next(iter( self.train_sampler ) )

		large_sample_x = torch.cat( [ s1X, s2X, s3X, s4X, s5X ], dim=0 )
		large_sample_y = torch.cat( [ s1Y, s2Y, s3Y, s4Y, s5Y ], dim=0 )

		self.network.initKFACStorage( large_sample_x.shape[0] )

		self.network.startRecording ()
		self.network.computeTempsForKFAC ( large_sample_x.cuda () )
		self.network.computeInitialKFACTerms ( large_sample_x.cuda () )
		self.network.stopRecording ( )

		self.network.initKFACStorage( self.network.batchSize )

		while (i <= self.params.max_iters): 

			start = datetime.datetime.now ()
			print( "Iteration %d "% (i) )

			self.network.setWeights( self.x )

			sampleX, sampleY = next (iter(self.train_sampler))
			X_sample_var = sampleX.type(torch.DoubleTensor).cuda ()
			Y_sample_var = sampleY.cuda ()

			self.network.zero_grad ()
			tr_ll, grad = self.network.computeGradientIter( X_sample_var, Y_sample_var  )
			self.stats.no_props =  sampleX.shape[0] 
			#print( "...done Gradient @ x")
			#print( "...done (train) LL: %e " % (tr_ll ) )
			#print( "... gradient norm --> %e " % (torch.norm( grad )) )

			'''
			for X, Y in self.test: 
				test_ll, test_accu = self.network.evalModel( X.cuda (), Y.cuda ())
			print( "...done (test) LL: %e and Accu: %3.2f " % (test_ll, test_accu) )
			print( "... starting TR Solver with approximated natural gradient")
			'''

			#KFAC here. 
			#compute F{-1} * gradient = approximated natural gradient here. 
			#convert the vector to structures. 
			self.network.startRecording ()
			self.network.computeTempsForKFAC ( X_sample_var )
			self.network.stopRecording ()
			x_kfac_t = self.network.computeKFACHv( X_sample_var, self.network.unpackWeights( grad ) )
			x_kfac = torch.cat([ w.contiguous().view( -1 ) for w in x_kfac_t ])
			x_kfac = Variable( x_kfac.type( torch.cuda.DoubleTensor ) )

			#compute eta * x_fac here. 
			# and use this to update the weights. 
			step = (delta) / torch.norm( x_kfac ).data
			x_kfac = Variable( torch.from_numpy( np.asarray( [ step ] ) ).type(torch.cuda.DoubleTensor) ) * x_kfac

			#compute the model reduction here. 
			# m = eta * grad * x_kfac + eta * eta * 0.5 * x_kfac * Hessian * x_kfac
			m_kfac = torch.dot( x_kfac, Variable( grad )) + 0.5 * torch.dot( self.network.computeHv( Variable( X_sample_var, requires_grad=True ), Variable( Y_sample_var ), x_kfac.data ), x_kfac )

			g_step = (delta) / torch.norm( grad )
			grad = g_step * grad

			m_g_kfac = torch.dot( Variable( grad.cuda () ), Variable( grad.cuda () ) ) + 0.5 * torch.dot( self.network.computeHv( Variable( X_sample_var, requires_grad=True ), Variable( Y_sample_var ), grad.cuda () ), Variable( grad.cuda () ) )

			print("Model Red: m_kfac: (%e, %e), g_kfac: (%e, %e)" % (m_kfac, torch.norm( x_kfac ), m_g_kfac, torch.norm(grad)) )

			self.network.zero_grad ()
			if (m_kfac.data[0] <= m_g_kfac.data[0]): 

				#for X, Y in self.train: 
				tr_ll, tr_accu = self.network.evalModel( X_sample_var, Y_sample_var )
				self.network.updateWeights( -x_kfac.data )

				#for X, Y in self.train: 
				self.network.zero_grad ()
				new_ll, new_accu = self.network.evalModel( X_sample_var, Y_sample_var )
				self.stats.no_props += len( self.train )
				rho = (tr_ll - new_ll) / (-m_kfac.data[0])

				if (rho.data[0] > 0.75) : #1e-4 
					self.x = self.x.add( -x_kfac.data )
					delta = min( self.params.max_delta, self.params.gamma1 * delta ) # 2
				elif( rho.data[0] >= 0.25 ): 
					self.x = self.x.add( -x_kfac.data )
				else: 
					delta /= self.params.gamma1 # 2

			else: 

				#for X, Y in self.train: 
				tr_ll, tr_accu = self.network.evalModel( X_sample_var, Y_sample_var )
				self.network.updateWeights( -grad.cuda () )

				#for X, Y in self.train: 
				self.network.zero_grad ()
				new_ll, new_accu = self.network.evalModel( X_sample_var, Y_sample_var )
				self.stats.no_props += len( self.train )
				rho = (tr_ll - new_ll) / (-m_g_kfac.data[0])

				if (rho.data[0] > 0.75) : #1e-4 
					self.x = self.x.add( -grad.cuda () )
					delta = min( self.params.max_delta, self.params.gamma1 * delta ) # 2
				elif( rho.data[0] >= 0.25 ): 
					self.x = self.x.add( -grad.cuda () )
				else: 
					delta /= self.params.gamma1 # 2

			X_sample_var.volatile = True
			Y_sample_var.volatile = True

			end = datetime.datetime.now ()
			print( ".... New LL: %e, rho: %e, old LL: %e, m_kfac: %e " % (new_ll, rho, tr_ll, -m_kfac) )
			print( "... TrustRegion Radius: %e " % (delta) )
			print( "... done with Iteration: %d in %f secs" % (i, (end - start).total_seconds ()) )
					
			#print stats here.
			self.stats.train_accu = 0
			self.stats.test_accu = 0

			self.stats.train_ll = new_ll
			self.stats.test_ll = 0

			self.stats.delta = delta
			self.stats.gradnorm = torch.norm( grad )

			self.stats.tr_failures = 0 
			self.stats.cg_iterations = 0 

			self.stats.iteration_time = ( end - start ).total_seconds ()
			self.stats.total_time += (end - start ).total_seconds ()

			self.stats.printIterationStats ()

			i += 1

		#cleanup and exit....
		self.stats.shutdown ()
