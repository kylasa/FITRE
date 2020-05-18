
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable


import numpy as np
import datetime

class SubsampledTRCGBFGS: 

	def __init__ (self, network, linsolver, params, stats, train, test, train_sampler, initialization, scale, bfgsObj ): 
		self.network = network
		self.train = train
		self.test = test
		self.params = params
		self.stats = stats
		self.linearSolver = linsolver
		self.train_sampler = train_sampler
		self.x = 0
		self.initialization = initialization
		self.scale = scale
		self.bfgsObj = bfgsObj

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
		self.network.updateWeights( self.x )
		
		prevWeights = self.x.clone ()

		for i in range( self.params.max_iters ): 

			start = datetime.datetime.now ()
			print( "Iteration %d "% (i) )

			tr_ll, grad, tr_accu  = self.network.computeGradientIter( self.train )
			self.stats.no_props = len( self.train.dataset )
			print( "...done Gradient @ x")
			print( "...done (train) LL: %e and Accu: %3.2f " % (tr_ll, tr_accu) )
			print( "... gradient norm --> %e " % (torch.norm( grad )) )

			test_ll, test_accu = self.network.evalModel( self.test )
			print( "...done (test) LL: %e and Accu: %3.2f " % (test_ll, test_accu) )

			print( "... starting TR Solver")

			sampleX, sampleY = next (iter(self.train_sampler))
			X_sample_var = Variable( sampleX.type(torch.DoubleTensor) .cuda () )
			Y_sample_var = Variable( sampleY.cuda () )

			prevWeights.copy_( self.x )

			failures = 0
			cg_iters = 0
			while True: 
				if failures == 0: 
					if self.initialization == 'zeros': 
						x_cg = self.getZeroVector ()
					else: 
						x_cg = self.getRandomVector ()
						x_cg = self.scale * x_cg * ( delta / torch.norm( x_cg ))
					x_cg = x_cg.cuda ()

				#get the dataset from the dataset. 
				#idx = np.random.randint( self.train.shape[0], size=2000 )
				#X_sample_var = Variable( sampleX )
	
				print( ".... starting CG")

				#x_cg, m, cg_iters, flag = self.linearSolver.solve( self.network.computeHv, x_cg, grad, delta, 250, 1e-8, X_sample_var, Y_sample_var, self.x)
				x_cg = self.bfgsObj.invHessVec( grad )	
				cg_iters = 0
				flag = 'None'

				#model reduction here. 
				# m = x. hess. x + x.grad
				m = 0.5 * torch.dot( x_cg, self.network.computeHv( X_sample_var, Y_sample_var, x_cg ).data ) + torch.dot( x_cg, grad )

				self.stats.no_props += cg_iters * 2 * self.params.sampleSize
				self.stats.no_mvp += cg_iters
				print("Model Red: %e, CG Iters: %d, Flag: %s, norm(X_CG): %e" % (m, cg_iters, flag, torch.norm( x_cg )) )

				self.network.updateWeights( x_cg )
				#new_ll, new_accu = self.network.evalModel( self.train )
				new_ll, new_grad, new_accu  = self.network.computeGradientIter( self.train )
				self.network.updateWeights( -x_cg )
				self.stats.no_props += len( self.train )

				rho = (tr_ll - new_ll) / (-m)
				print( ".... New LL: %e, rho: %e " % (new_ll, rho) )


				#if (m >= 0) or (rho < self.params.eta2) : 
				if (rho < self.params.eta2) : 
					failures += 1
					delta /= self.params.gamma1
					x_cg = x_cg * ( delta / torch.norm( x_cg ) )
					print( ".... Failure: %d " % (failures) )
				elif rho < self.params.eta1: 
					self.x += x_cg
					self.network.updateWeights( x_cg )
					delta = min( self.params.max_delta, self.params.gamma2 * delta )
					print( "... SUPER SUCCESS: " )
					break
				else: 
					self.x += x_cg
					self.network.updateWeights( x_cg )
					delta = min( self.params.max_delta, self.params.gamma1 * delta )
					print( "... SUCCESS: " )
					break

			X_sample_var.volatile = True
			Y_sample_var.volatile = True

			if failures >= 100: 
				print( ".... No failures is more than expected... exiting... \n")
				exit ()

			#update the bfgs updater now.... 
			s = self.x - prevWeights
			y = new_grad - grad
			self.bfgsObj.update( s, y )

			end = datetime.datetime.now ()
			print( "... done with Iteration: %d in %f secs" % (i, (end - start).total_seconds ()) )
					
			#print stats here.
			self.stats.train_accu = tr_accu
			self.stats.test_accu = test_accu

			self.stats.train_ll = tr_ll
			self.stats.test_ll = test_ll

			self.stats.delta = delta
			self.stats.gradnorm = torch.norm( grad )

			self.stats.tr_failures = failures
			self.stats.cg_iterations = cg_iters

			self.stats.iteration_time = ( end - start ).total_seconds ()
			self.stats.total_time += (end - start ).total_seconds ()

			self.stats.printIterationStats ()

		#cleanup and exit....
		self.stats.shutdown ()
