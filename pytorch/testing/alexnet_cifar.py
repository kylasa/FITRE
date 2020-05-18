
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

class AlexNetCIFAR(nn.Module):

	def __init__(self, num_classes=10 ):
		super(AlexNetCIFAR, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=5, stride=2, padding=1)
		self.conv2 = nn.Conv2d(in_channles = 64, out_channels = 64, kernel_size = 5, stride = 2, padding = 1)

		self.m1 = nn.MaxPool2d( kernel_size = 3, stride = 2, padding = 1 )
		self.m2 = nn.MaxPool2d( kernel_size = 3, stride = 2, padding = 1 )

		self.c1 = Swish ()       
		self.c2 = Swish ()       

		self.l1 = Swish ()       
		self.l2 = Swish ()       

		self.fc1   = nn.Linear(64 * 8 * 8, 384)
		self.fc2   = nn.Linear(384, 192) 
		self.fc3   = nn.Linear(192, 10) 

		self.offsets = [0]
		self.activation = activation

	def forward(self, x): 
		out = self.conv1( x )
		out = self.c1 ( out )
		out = self.m1( out )

		out = self.conv2( out )
		out = self.c2( out )
		out = self.m2( out )

		out = self.fc1( out )
		out = self.l1( out )

		out = self.fc2( out )
		out = self.l2( out )

		out = self.fc3( out )

		return out

	def writeFile( self, params ):  
		arr = []
		for p in params: 
			nparr = p.data.cpu ().numpy ()
			print (nparr.shape)
			arr.append( np.reshape( nparr, nparr.shape, order='F' ).ravel ()) 

		weights = np.concatenate( arr )
		print( len( weights ) )

		with open( 'lenet_kaiming.txt', 'w') as f:
			for w in weights:
				f.writelines("%3.10f\n" % w )


    def initOffsets (self): 
        for W in self.parameters (): 
            self.offsets.append( W.numel () )
        self.offsets = np.cumsum( self.offsets )

    def getParamLen( self ): 
        l = 0
        for W in self.parameters (): 
            print (W.numel ())
            l += W.numel ()
        return l

    def setLossFunction( self, ll ): 
        self.lossFunction = ll

    def _computeGradient( self, func, create_graph): 
        if create_graph: 
            g = autograd.grad( func, self.parameters (), create_graph=True )
        else:
            g = autograd.grad( func, self.parameters (), create_graph=False )
        return g

    def computeGradientIter( self, X, Y, regularization): 

        self.zero_grad ()
        x_var = Variable( X )
        y_var = Variable( Y )

        loss, pred = self.evalModelChunk( x_var, y_var )
        accu = pred.eq( y_var ).type(torch.LongTensor).sum ()
        self.zero_grad ()
        g = self._computeGradient( loss, False )

        x_var.volatile = True
        y_var.volatile = True

        return g
        
    def computeHv( self, sampleX, sampleY, vec, regularization ): 
        self.zero_grad ()
        loss, pred = self.evalModelChunk( sampleX, sampleY )
        gradient = self._computeGradient( loss, True )
        self.zero_grad ()
        hv = self._computeGradient( (Variable( vec ) * gradient).sum (), False )

        W = self.getWeights ()
        return hv / sampleX.shape[0] + regularization * Variable( vec )

    def evalModel( self, X, Y): 
        x_var = Variable( X, requires_grad=True )
        y_var = Variable( Y  )

        out = self( x_var )
        loss = self.lossFunction( out, y_var )
        accu = (torch.max( out, 1)[1].eq( y_var ) ).type(torch.cuda.LongTensor).sum ()

        x_var.volatile=True
        y_var.volatile=True

        return loss


    def evalModelChunk( self, x, y ): 
        out = self (x)
        loss = self.lossFunction( out, y )
        return loss, torch.max( out, 1)[1]

    def initWeightsMatrix( self, w ): 
        idx = 0
        for W in self.parameters (): 
            W.data.copy_( w[ idx ] )
            idx += 1

    def initRandomWeights (self ): 
        for W in self.parameters (): 
            W.data.uniform_(0, 1)
            W.data *= 0.1

    def initZeroWeights( self ): 
        for W in self.parameters (): 
            W.data.fill_(0)

    def initWeights( self, val ): 
        for W in self.parameters (): 
            W.data.fill_(val)

    def setWeights( self, vec ): 
        idx = 0
        for W in self.parameters ():
           W.data.copy_( torch.index_select( vec, 0, torch.arange( self.offsets[idx], self.offsets[ idx ] + W.numel () ).type( torch.cuda.LongTensor ) ).view( W.size () ) )
           idx += 1

    def setWeightsBiases( self, c1, c2, c3, c4, l1, l2, l3, l4, l5, l6): 
        self.conv1.weight.data.copy_( c1 )
        self.conv1.bias.data.copy_( c2 )
        self.conv2.weight.data.copy_( c3 )
        self.conv2.bias.data.copy_( c4 )

        self.fc1.weight.data.copy_( l1 )
        self.fc1.bias.data.copy_( l2 )
        self.fc2.weight.data.copy_( l3 )
        self.fc2.bias.data.copy_( l4 )
        self.fc3.weight.data.copy_( l5 )
        self.fc3.bias.data.copy_( l6 )

    def updateWeights( self, vec ): 
        idx = 0
        for W in self.parameters ():
           W.data.add_( torch.index_select( vec, 0, torch.arange( self.offsets[idx], self.offsets[ idx ] + W.numel () ).type( torch.cuda.LongTensor ) ).view( W.size () ) )
           idx += 1

    def getWeights( self ): 
        return torch.cat([ w.contiguous().view( -1 ).data.clone () for w in self.parameters () ])

    def initXavierUniform( self ): 
        for W in self.parameters (): 
            if len(W.data.size () ) > 1: 
                nn.init.xavier_uniform( W.data ) 
            else: 
                W.data.random_(0, 4)
                W.data *= 0.1

    def initKaimingUniform( self ): 
        for W in self.parameters (): 
            if len(W.data.size () ) > 1: 
                nn.init.kaiming_uniform( W.data ) 
            else: 
                W.data.random_(0, 4)
                W.data *= 0.1

			self.writeFile( self.network.parameters () )

