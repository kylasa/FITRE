
import torch

class BFGSUpdate: 

	def __init__ (self, numVecs, c): 
		self.idx= 0
		self.done = False
		self.maxVectors = numVecs

		self.S = torch.zeros( c, numVecs ).type( torch.DoubleTensor ).cuda ()
		self.Y = torch.zeros( c, numVecs ).type( torch.DoubleTensor ).cuda ()
		self.alpha = torch.zeros( numVecs ).type( torch.DoubleTensor ).cuda ()

	def update(self, s, y ): 
		#print( 'dot product: %e, at i=%d, %e, %e' % (torch.dot( s, y), self.idx, torch.norm(s), torch.norm(y))) 
		#if (not self.done) and (self.idx != 0): 
		#if (not self.done) and (self.idx != 0): 
		#copy here... do not create more memory here. 
		self.S[ :, self.idx ] = s
		self.Y[ :, self.idx ] = y 
		self.idx = (self.idx + 1) % self.maxVectors

		if (self.idx == 0): 
			self.done = True

	#as described in page. 178
	def invHessVec(self, vec ): 

		q = vec.clone ()
		
		#reverse traversal here. 
		if self.idx != 0: 
			for i in range( self.idx-1, -1, -1 ):  #from idx-1 to 0
				s = self.S[ :, i ]
				y = self.Y[ :, i ]
				sy = torch.dot( s, y )
				#print( 'Dot product for %d is %e' % (i, sy) )
				self.alpha[ i ] = (1./sy) * torch.dot( s, q )	
				q = q - self.alpha[ i ] * y

		if self.done: 
			for i in range( self.maxVectors-1, self.idx - 1, -1 ): # from max to idx	
				s = self.S[ :, i ]
				y = self.Y[ :, i ]
				sy = torch.dot( s, y )
				#print( 'Dot product for %d is %e' % (i, sy) )
				self.alpha[ i ] = (1./sy) * torch.dot( s, q )	
				q = q - self.alpha[ i ] * y


		#forward travel now... 
		r = q

		if self.done: 
			for i in range( self.idx, self.maxVectors ): # from idx to maxVecs(excluding)
				s = self.S[ :, i ]
				y = self.Y[ :, i ]
				sy = torch.dot( s, y )
				#print( 'Dot product for %d is %e' % (i, sy) )

				beta = (1.0 / sy) * torch.dot( y, r )
				r = r + (self.alpha[ i ] - beta ) * s

		for i in range( self.idx ): # from 0 to idx-1
			s = self.S[ :, i ]
			y = self.Y[ :, i ]
			sy = torch.dot( s, y )
			#print( 'Dot product for %d is %e' % (i, sy) )

			beta = (1.0 / sy) * torch.dot( y, r )
			r = r + (self.alpha[ i ] - beta ) * s
			
		return r
