
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from utils import group_product
from utils import group_add


import numpy as np
import datetime
import math
import os
import readWeights

FILE_NAME = 'alexnet_xavier.txt'
TYPE = torch.FloatTensor

class SubsampledTRCGKFACFULL: 

	def __init__ (self, network, linsolver, params, stats, train, test, train_test, train_sampler, momentum, check_grad, debug ): 
		self.network = network
		self.train = train
		self.test = test
		self.train_test = train_test
		self.params = params
		self.stats = stats
		self.linearSolver = linsolver
		self.train_sampler = train_sampler
		self.momentum = momentum
		self.check_grad = check_grad
		self.debug = debug

	def writeFile( self, params ): 
		arr = []
		for p in params: 
			nparr = p.data.cpu ().numpy ()
			print (nparr.shape)
			arr.append( np.reshape( nparr, nparr.shape, order='F' ).ravel ())

		weights = np.concatenate( arr )
		print( len( weights ) )

		with open( FILE_NAME, 'w') as f: 
			for w in weights: 
				f.writelines("%3.10f\n" % w )

	def readFile( self, params ): 

		if os.path.exists( FILE_NAME ): 
			temp = readWeights.readMatrix( FILE_NAME, [ [64, 3, 5, 5], [64], [64, 64, 5, 5], [64], [384, 4096], [384], [192, 384], [192], [10, 192], [10] ] )
			#temp = readWeights.readMatrix( FILE_NAME, [ [6, 3, 5, 5], [6], [16, 6, 5, 5], [16], [120, 400], [120], [84, 120], [84], [10, 84], [10] ] )

			for p, t in zip( params, temp): 
				p.data.copy_( torch.from_numpy( t ) )
			return True
		else: 
			False
		

	def initializeZero( self ): 
		for p in self.network.parameters ():
			p.data.fill_( 0 )

	def initializeRandom( self ): 
		x = [ torch.randn( W.numel() ).type( TYPE ) for W in self.network.parameters () ]

	def initFromPeng( self ): 
		if not self.readFile( self.network.parameters () ):
			self.network.initFromPeng()
			self.writeFile( self.network.parameters () )

	def initXavier( self ): 
		if not self.readFile( self.network.parameters () ):
			self.network.initXavierUniform()
			self.writeFile( self.network.parameters () )

	def initKaiming( self ): 
		if not self.readFile( self.network.parameters () ):
			self.network.initKaimingUniform()
			self.writeFile( self.network.parameters () )

	def initHybrid( self ): 
		self.network.initHybrid ()
	
	def initConstant( self ): 
		self.network.initconstant ()

	def getZeroVector (self): 
		z = [ torch.zeros( W.numel() ).type( TYPE ) for W in self.network.parameters () ]
		return torch.cat( [ p.contiguous().view(-1) for p in z ] ).cuda ()

	def getRandomVector (self): 
		r = [ torch.randn( W.numel() ).type( TYPE ) for W in self.network.parameters () ]
		return ( torch.cat( [ p.contiguous().view(-1) for p in r ] )).cuda ()


	def solve( self ): 

		i = 0
		failures = 0

		train_set = self.moveToDevice( self.train )
		test_set = self.moveToDevice( self.test )
		#train_test = self.moveToDevice( self.train_test )
		self.network.initKFACStorage( )

		self.prevWeights = []
		for p in self.network.parameters (): 
			self.prevWeights.append( torch.zeros_like( p ) )

		while (i <= self.params.max_iters): 

			start = datetime.datetime.now ()

			# Make one full pass of the dataset here. 
			#self.fullPass (self.train)
			self.fullPass( train_set, i )

			end = datetime.datetime.now ()

			# Compute Train and Test Accuracy here. 
			#train_ll, tr_accu = self.getStatistics2 ( self.train_test)
			#test_ll, test_accu = self.getStatistics2 ( self.test)
			train_ll = 0
			tr_accu = 0
			test_ll, test_accu = self.getStatistics2 (test_set )
			
					
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

	def getStatistics2(self, test_loader ):
		#model.eval()
		test_loss = 0 
		correct = 0 
		count = 0
		for data, target in test_loader:
			data, target = data.type( TYPE ).cuda(), target.type(torch.LongTensor).cuda()
			output = self.network(data)
			test_loss += self.network.lossFunction(output, target).item()*data.size()[0]  # sum up batch loss
			pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()
			count += len( data )

		print( count )
		test_loss /= count
		acc = 100.0 * correct.item() / count
		#model.train()
		return test_loss, acc 

	def moveToDevice (self, dataset ): 
		l = []
		for x, y in dataset: 
			l.append( (x.type(TYPE).cuda(), y.cuda()) )
		return l


	def fullPass( self, train, curIteration ): 

		offset = 0
		for l in train: 

			print ()
			print( 'Iteration: %d, Offset: %d' % (curIteration, offset ))
			torch.cuda.empty_cache ()
			#torch.no_grad ()
			if (self.debug):
				print ()
				print ()
				print ()
				print( 'Starting the full pass... ')
				print( l[0][0, 0, :, : ] )
				print( l[0][0, 1, :, : ] )
				print( l[0][0, 2, :, : ] )
				print ()
				print ()
				print ()

				print( l[1] )

				print( l[0][255, 0, :, : ] )
				print( l[0][255, 1, :, : ] )
				print( l[0][255, 2, :, : ] )

			X_sample_var = l[0].type( TYPE ).cuda ()
			Y_sample_var = l[1].type( torch.LongTensor ).cuda ()


			# With regularization
			self.network.zero_grad ()
			grad = self.network.computeGradientIter2( X_sample_var, Y_sample_var  )

			#if (self.debug):
			print( 'grad norm', math.sqrt( group_product( grad, grad ) ) )
			print( 'weights norm', math.sqrt( group_product( self.network.parameters (), self.network.parameters () ) ) )

			#for p in self.network.parameters (): 
			#	print( 'layer Norm: ', torch.norm( p.grad.data ) )

			#print( 'conv1: ', self.network.conv1.bias.grad.data )
			#print( 'conv2: ', self.network.conv2.bias.grad.data )
				#import pdb;pdb.set_trace();
				
			tr_ll, tr_accu = self.network.evalModel( X_sample_var, Y_sample_var )
			#print( grad )
			#print( 'Model: ', tr_ll )

			#KFAC here. 
			#compute F{-1} * gradient = approximated natural gradient here. 
			#convert the vector to structures. 
			self.network.zero_grad ()
			#print( 'grad norm', math.sqrt( group_product( grad, grad ) ) )
         
			self.network.startRecording ()
			self.network.computeTempsForKFAC ( X_sample_var )
			self.network.stopRecording ()

			x_kfac = self.network.computeKFACHv( X_sample_var, grad )
			print( 'Norm of x_kfac: ', math.sqrt( group_product( x_kfac, x_kfac ) ) )

			# assert here, KFAC is PSD
			if (group_product( grad, x_kfac ) < 0 ): 
				print( 'Trouble with KFAC direction... it is NOT PSD ' )
				exit ()

			#compute the model reduction here. 
			# m = eta * grad * x_kfac + eta * eta * 0.5 * x_kfac * Hessian * x_kfac
			# with regularization
			Hv = self.network.computeHv_r( X_sample_var, Y_sample_var, x_kfac )
			#Hv = self.network.computeHv( X_sample_var, Y_sample_var, x_kfac )
			print( 'Norm of Hv: ', math.sqrt( group_product( Hv, Hv ) ) )
			#s = 0
			#for l in Hv: 
			#	s += torch.norm( l ) * torch.norm( l )
			#print( s, torch.sqrt( s ) )

			vHv = group_product( Hv, x_kfac ).item ()
			print( 'KFAC : vHv ', vHv )

			vnorm = math.sqrt( group_product( x_kfac, x_kfac ) )
			if (self.debug):
				print( 'KFAC Norm: ', vnorm )
			#handle Negative Curvature here. 
			if ( vHv < 0 ): 
				#step = (self.params.delta) / vnorm
				#x_kfac = [ v * step for v in x_kfac ]
				#m_kfac = vHv * 0.5 * step * step - group_product( grad, x_kfac ).item ()
				x_kfac = [ v * self.params.delta / vnorm for v in x_kfac ]
				#print( 'grad * x_kfac: ', group_product( grad, x_kfac ).item () )
				#print( 'vHv term : ', vHv * 0.5 * self.params.delta * self.params.delta / vnorm / vnorm )
				m_kfac = vHv * 0.5 * self.params.delta * self.params.delta / vnorm / vnorm - group_product( grad, x_kfac ).item ()
				print( 'Model Reduction kfac direction (Negative): ', m_kfac )
				if (self.debug):
					print( 'alpha (NC): ', self.params.delta / vnorm )
			else: 
				#import pdb;pdb.set_trace();
				gv = group_product( x_kfac, grad ).item ()
				if (self.debug):
					print( 'group product: ', gv )
				step = gv / (vHv + 1e-6)
				step = min( step, (self.params.delta / (vnorm + 1e-16)) )
				x_kfac = [ v * step for v in x_kfac ]
				if (self.debug):
					print( 'alpha ', step )
				m_kfac = vHv * 0.5 * step * step - gv * step
				print( 'Model Reduction kfac direction (PSD) : ', m_kfac )

				
			if (self.check_grad == True): 
				self.network.zero_grad ()

				grad_dot = group_product( grad, grad ).item ()
				#print( 'Grad norm: ', math.sqrt( grad_dot ) )
				t = self.network.computeHv_r( X_sample_var, Y_sample_var, grad )
				print( 'Grad Hv Norm: ', math.sqrt( group_product( t, t ) ) )
				#print( 'Grad v horm: ', math.sqrt( group_product( grad, grad ).item ()) )

				vHv = group_product ( t, grad ).item ()
				#print( 'Grad: vHv ', vHv )
				print( 'Grad v horm: ', math.sqrt( group_product( grad, grad ).item ()) )
				vnorm = math.sqrt( grad_dot )
				
				if (vHv < 0 ): 
					#step = (self.params.delta) / vnorm
					#grad = [ g * step for g in grad ]
					#m_g_kfac = 0.5 * step * step * vHv - group_product( grad, grad )
					#print( 'Grad grad * x_kfac: ', group_product( grad, grad).item () )
					m_g_kfac = vHv * 0.5 * self.params.delta * self.params.delta / vnorm / vnorm - group_product( grad, grad ).item ()  * self.params.delta / vnorm
					grad = [ v * self.params.delta / vnorm for v in grad ]
		
					#print( 'Grad grad * x_kfac: ', group_product( grad, grad).item () )
					#print( 'Grad vHv term : ', vHv * 0.5 * self.params.delta * self.params.delta / vnorm / vnorm )
					#print( 'Grad NC alpha:  ', self.params.delta/vnorm)
					print( 'Model reduction gradient kfac( negative ): ', m_g_kfac )
				else: 
					gv = grad_dot
					step = gv / (vHv + 1e-6)
					#print( 'Grad alpha: ', step )
					step = min( step, self.params.delta / (vnorm + 1e-16) )
					grad = [ g * step  for g in grad ]
					m_g_kfac = vHv * 0.5 * step * step - gv * step
					print( 'Model reduction gradient kfac( psd) ): ', m_g_kfac )

			#print( m_g_kfac, m_kfac )
			#import pdb;pdb.set_trace();

			if ( (not self.check_grad ) or (m_kfac < m_g_kfac ) ): 
				#use Natural Gradient	
				#Momentum Here
				#Momentum
				for idx, p in enumerate( self.prevWeights ): 
					x_kfac[idx].add_ (self.momentum, p )
					
				#self.network.updateWeights( x_kfac.data.mul_( kfac_step.item () ) )
				for idx, w in enumerate( self.network.parameters () ): 
					w.data.add_( -1., x_kfac[ idx ] )
					self.prevWeights[ idx ].copy_( x_kfac[ idx ] )

				new_ll, new_accu = self.network.evalModel( X_sample_var, Y_sample_var )

				rho = (new_ll - tr_ll) / ( m_kfac - 1e-16 )
				if (self.debug): 
					print( rho )

			else: 
				#tr_ll, tr_accu = self.network.evalModel( X_sample_var, Y_sample_var )

				#grad.add_ (self.momentum, self.prevWeights )
				#group_add( grad, self.prevWeights, self.momentum )
				for v, mom in zip( grad, self.prevWeights ): 
					v.data.add_( self.momentum, mom )

				#self.network.updateWeights( grad.data.mul_( grad_step.item () ) )
				#self.prevWeights.copy_( grad.data )
				for idx, w in enumerate( self.network.parameters () ): 
					w.data.add_( -1., grad[ idx ] )
					self.prevWeights[ idx ].copy_( grad[ idx ] )

				new_ll, new_accu = self.network.evalModel( X_sample_var, Y_sample_var )
				rho = (new_ll - tr_ll) / (m_g_kfac - 1e-16)

			if (rho > 0.75) : #1e-4 
				self.params.delta = min( self.params.max_delta, 2. * self.params.delta ) # 2
			if (rho < 0.25): 
				self.params.delta = max( self.params.min_delta, 0.5 * self.params.delta ) # 2
			if ((rho < 1e-4) or (new_ll > (10. * tr_ll))): 
				if(self.debug): 
					print( 'Trouble.... Reject this step, since there  is no VISIBLE decrease ' )
				#self.network.updateWeights( -1 * self.prevWeights ); 
				for idx, w in enumerate( self.network.parameters () ):
					w.data.add_ (1., self.prevWeights[ idx ] )
			if (self.debug): 
				#print( m_g_kfac, m_kfac )
				if ( (not self.check_grad ) or (m_kfac < m_g_kfac ) ): 
					print( rho, self.params.delta, new_ll, tr_ll, m_kfac)
				else: 
					print( rho, self.params.delta, new_ll, tr_ll, m_g_kfac)

			if ( (not self.check_grad ) or (m_kfac < m_g_kfac ) ): 
				print( '%4.10e  %4.10e  %4.10e  %4.10e %4.10e  %6s  %3.6e' %(tr_ll, new_ll, rho, m_kfac, math.sqrt( group_product( self.network.parameters (), self.network.parameters () ) ), 'kfac', self.params.delta ) )
			else: 
				print( '%4.10e  %4.10e  %4.10e  %4.10e %4.10e  %6s  %3.6e' %(tr_ll, new_ll, rho, m_g_kfac, math.sqrt( group_product( self.network.parameters (), self.network.parameters () ) ), 'grad', self.params.delta ) )

			del grad
			offset += l[1].size()[0]

