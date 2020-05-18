
import torch
import numpy as np

# pk			: Natural Gardient
# gradient	: Gradient
# Hv			: Hessian Function Pointer
# epsilon	: Machine tolerance

#t				: is the step size to be computed

def getStepModelReduction( t, vk, rk, hrk ): 
	return -t * torch.dot( vk, rk ) + 0.5 * t * t * torch.dot( rk, hrk )

def getFrankWolfeUpdate(pk, gradient, network, delta, maxiters, epsilon, X, Y): 

	tk = 0
	for i in range( maxiters ): 
		print( 'Iteration %d of Frank update', i )
		import pdb;pdb.set_trace();
		vk = gradient + network.computeHv( X, Y, pk )
		qk = -( delta / torch.norm( vk ).data[0] ) * vk

		first_opt = torch.dot( pk - qk, vk )
		if (first_opt <= epsilon) : 
			break
	
		rk = pk - qk
		hrk = network.computeHv( X, Y, rk )
		
		h0 = 0
		h1 = getStepModelReduction( 1 )
		if (torch.dot( rk, hrk ) <= 0 ): 

			if (h0 <= h1): 
				tk = 0 
			else: 
				tk = 1

		else: 
			tk = torch.dot( vk, rk ) / torch.dot( rk, hrk )
			ht = getStepModelReduction( tk )

			index = np.argmax( [h0, h1, ht] )
			tk = [0, 1, tk][ index ]

		pk = pk - tk * ( pk - qk )

	gv = torch.dot( gradient, pk )
	hv = hv( pk )
	m =  gv + 0.5 * torch.dot( pk, hv )

	return pk, m
