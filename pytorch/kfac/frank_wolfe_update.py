
import torch
import numpy as np
import math

from utils import group_product
from utils import group_add

# pk			: Natural Gardient
# gradient	: Gradient
# Hv			: Hessian Function Pointer
# epsilon	: Machine tolerance

#t				: is the step size to be computed

def getStepModelReduction( t, vk, rk, hrk ): 
	return -t * group_product( vk, rk ) + 0.5 * t * t * group_product( rk, hrk )

def getFrankWolfeUpdate(pk, gradient, network, delta, maxiters, epsilon, X, Y): 

	tk = 0
	for i in range( maxiters ): 

		print( 'Iteration %d of Frank update'% ( i ) )

		# Scale P, so that it would be in the trust region radius = Delta
		pk = delta / math.sqrt( group_product( pk, pk ) ) 

		vk = [ (g + p) for g, p in zip( gradient, network.computeHv_r( X, Y, pk)) ]
		vnorm = math.sqrt( group_product( vk, vk ) )
		qk = [v.mul_( -delta / vnorm )  for v in vk ]

		rk = group_add( pk, qk, -1 )
		rk = [ r.type( torch.cuda.DoubleTensor ) for r in rk ]

		first_opt = group_product( rk, vk )
		if (first_opt.item () <= epsilon) : 
			print( 'first_opt for ', first_opt.item () )
			break
	
		hrk = network.computeHv_r( X, Y, rk )
		
		h0 = 0
		h1 = getStepModelReduction( 1, vk, rk, hrk )
		if (group_product ( rk, hrk ) <= 0 ): 

			if (h0 <= h1): 
				tk = 0 
			else: 
				tk = 1

		else: 
			tk = (group_product( gradient, rk ) + group_product( pk, hrk )) / group_product( rk, hrk )
			ht = getStepModelReduction( tk, vk, rk, hrk )

			index = np.argmax( [h0, h1, ht] )
			tk = [0, 1, tk][ index ]

		pk = [ (p - tk * q) for p, q in zip( pk, group_add( pk, qk, -1) ) ]

	gv = group_product( gradient, pk )
	hv = network.computeHv_r( X, Y, pk )
	m =  gv + 0.5 * group_product( pk, hv )

	return pk, m
