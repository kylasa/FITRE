
import torch
import numpy as np
import math

class ConjugateGradient: 

	@staticmethod
	def solve( fptr, x, b, delta, maxIter, tolerance, dataX, dataY, curSol): 
	
		g = b.clone ()
		hv = fptr( dataX, dataY, x).data
		r = -1.0 * g - fptr( dataX, dataY, x ).data
		#print( "Norm of residual: %e " % (torch.norm( r )))
		#print( "Norm of B: %e " % (torch.norm( b )) )
		#print( "Norm of x: %e " % (torch.norm( x )) )
		#print( "hv: %e " % (torch.norm( hv )) )

		z = r
		rho = torch.dot( z, r )
		#print( "rho: %e " % (rho) )

		tst = torch.norm( r )
		terminate = tolerance * tst
		it = 1
		hat_del = delta

		rho_old = 1.0

		flag = "We do not know"
		#print( tst, terminate, it, maxIter, torch.norm(x), hat_del )

		if (tst <= terminate ): 
			iflag = 'Small ||g||'

		while (tst > terminate) and (it <= maxIter) and (torch.norm(x) <= hat_del): 
			if it == 1: 
				p = z
			else: 
				beta = rho / rho_old
				p = z + beta * p

			w = fptr( dataX, dataY, p).data
			alpha = torch.dot( p, w )
			#print( "alpha: %e, hessianvec: %e " % (alpha, torch.norm( w )) )

			if (alpha <= 0): 
				ac = torch.dot( p, p )
				bc = 2.0 * torch.dot( x, p )
				cc = torch.dot( x, x ) - delta * delta

				alpha = (-bc + math.sqrt( bc * bc - 4.0 * ac * cc )) / (2. * ac)
				flag = "Negative Curvature"
				x = x + alpha * p

				#print( "ac: %e " % (ac) )
				#print( "bc: %e " % (bc) )
				#print( "cc: %e " % (cc) )
				#print( "xx: %e " % (torch.dot(x,x)) )
				#print( "dd: %e " % (delta * delta) )
				#print( "alpha of NC: %e" % ( alpha ) )

				break
			else:
				alpha = rho / alpha
				if torch.norm( x + alpha * p ) > delta: 
					ac = torch.dot( p, p )
					bc = 2.0 * torch.dot( x, p )
					cc = torch.dot( x, x ) - delta * delta

					alpha = (-bc + math.sqrt( bc * bc - 4.0 * ac * cc )) / (2. * ac)
					flag = "Boundry Condition"
					x = x + alpha * p
					break

			x = x + alpha * p
			r = r - alpha * w

			tst = torch.norm( r )
			if tst <= terminate:
				flag = " || r || < test "
				break
			if torch.norm( x ) >= hat_del: 
				flag = " Close to boundry"
				break

			rho_old = rho
			z = r
			rho = torch.dot( z, r )
			it += 1

		num_cg = it

		# Evaluate the tr model here
		m = 0.5 * torch.dot( x, fptr( dataX, dataY, x ).data ) + torch.dot( x, b )

		return x, m, num_cg, flag
