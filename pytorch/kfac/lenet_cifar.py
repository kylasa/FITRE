
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

class Swish(nn.Module):
    """Swish Function: 
    Applies the element-wise function :math:`f(x) = x / ( 1 + exp(-x))`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = Swish()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return input * torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class LeNetCIFAR(nn.Module):

    def __init__(self, num_classes=10, activation='relu'):
        super(LeNetCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84) 
        self.fc3   = nn.Linear(84, 10) 
        self.offsets = [0]
        self.activation = activation

    def forward(self, x): 
        out = Swish(self.conv1(x))
        out = F.avg_pool2d(out, 2)
        out = Swish(self.conv2(out))
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1) 
        out = Swish(self.fc1(out))
        out = Swish(self.fc2(out))
        out = self.fc3(out)
        return out

    def initOffsets (self): 
        for W in self.parameters (): 
            self.offsets.append( W.numel () )
        self.offsets = np.cumsum( self.offsets )

    def setLossFunction( self, ll ): 
        self.lossFunction = ll

    def _computeGradient( self, func, create_graph): 
        if create_graph: 
            g = autograd.grad( func, self.parameters (), create_graph=True )
        else:
            g = autograd.grad( func, self.parameters (), create_graph=False )
        return torch.cat( [ grad.contiguous ().view( -1 ) for grad in g if grad is not None ] )

    def _computeGradBackward( self, func ): 
        func.backward ()

    def computeGradientIter( self, data ): 
        g = 0
        ll = 0
        cum_loss = 0
        accu = 0
        counters = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for X, Y in data: 

            self.zero_grad ()
            x_var = Variable( X.type( torch.DoubleTensor ).cuda () )
            y_var = Variable( Y.type( torch.LongTensor ).cuda () )

            lll, loss, pred = self.evalModelChunk( x_var, y_var )
            ll += lll
            cum_loss += loss
            accu += (pred.eq( y_var )).cpu ().type(torch.LongTensor).sum ().data[0]

            for a, b in zip( pred.cpu ().data, Y ): 
                if a == b: 
                    counters[ a ] += 1

            g += self._computeGradient( loss, False )
            x_var.volatile = True
            y_var.volatile = True

        print (accu)
        print ( counters )
        #return (ll.cpu().data[0] / len( data.dataset ) ), (cum_loss.cpu ().data[0] / len( data.dataset )), (g.data / len( data.dataset )), np.float(accu) * 100. / np.float(len( data.dataset ))
        return (cum_loss.cpu ().data[0] / len( data.dataset )), (g.data / len( data.dataset )), np.float(accu) * 100. / np.float(len( data.dataset ))
        
    def computeHv( self, sampleX, sampleY, vec ): 
        ll, loss, pred = self.evalModelChunk( sampleX, sampleY )
        self.zero_grad ()
        gradient = self._computeGradient( loss, True )
        hv = self._computeGradient( (Variable( vec ) * gradient).sum (), False )
        return hv / len( sampleY )

    def initWeightsMatrix( self, w ): 
        idx = 0
        for W in self.parameters (): 
            W.data.copy_( w[ idx ] )
            idx += 1

    def initRandomWeights (self ): 
        for W in self.parameters (): 
            W.data.uniform_(0, 1)

    def initZeroWeights( self ): 
        for W in self.parameters (): 
            W.data.fill_(0)

    def initWeights( self, vec ): 
        idx = 0
        for W in self.parameters ():
           W.data.copy_( torch.index_select( vec, 0, torch.arange( self.offsets[idx], self.offsets[ idx ] + W.numel () ).type( torch.cuda.LongTensor ) ).view( W.size () ) )
           idx += 1

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
   

    def evalModel( self, dataIter ): 
        value = 0
        ll = 0
        accu = 0

        for data, label in dataIter: 
            x_var = Variable( data.type( torch.DoubleTensor ).cuda () )
            y_var = Variable( label.type( torch.LongTensor ).cuda () )
            out = self( x_var )
            cur_loss = self.lossFunction( out, y_var )
            value += cur_loss
            ll += cur_loss
            accu += (torch.max( out, 1)[1].eq( y_var ) ).cpu ().type(torch.LongTensor).sum ().data[0]

            x_var.volatile=True
            y_var.volatile=True

        print( accu )
        #return ( ll.cpu ().data[0] / len( dataIter.dataset) ),  (value.cpu().data[0] / len( dataIter.dataset )), np.float( accu )  * 100. / np.float( len( dataIter.dataset ) )
        return (value.cpu().data[0] / len( dataIter.dataset )), np.float( accu )  * 100. / np.float( len( dataIter.dataset ) )

    def evalModelChunk( self, x, y ): 
        out = self (x)
        loss = self.lossFunction( out, y )
        return loss, loss, torch.max( out, 1)[1]

    def runDerivativeTest( self, numPoints, x, noise, dataIter ): 

        dx = [] #norm of the noise
        dx_2 = [] # diff squared
        dx_3 = [] #diff cubed
        c = noise

        for i in range( numPoints ): 
            
            fr = 0
            fs = 0    
            fs_grad = 0
            fs_hessian = 0

            b = [ (w1 + w2) for w1, w2 in zip(x, c) ]
            self.initWeightsMatrix( b )

            cv = [ Variable(W, requires_grad=True) for W in c ]  
            flat_random = torch.cat( [ p.contiguous().view( -1 ) for p in cv ] )
            ll, fr, accu = self.evalModel( dataIter )

            for X, Y in dataIter: 

                x_var = Variable( X.cuda () )
                y_var = Variable( Y.cuda () )

                self.zero_grad ()
                self.initWeightsMatrix( x )
                ll, fs_loss, accu = self.evalModelChunk( x_var, y_var )

                gs = self._computeGradient( fs_loss, True )
                first_order = torch.dot( flat_random, gs )

                hv = self._computeGradient( (flat_random * gs).sum (), False )
                second_order = 0.5 * torch.dot( flat_random, hv )

                x_var.volatile = True
                y_var.volatile = True

                fs += fs_loss 
                fs_grad += first_order
                fs_hessian += second_order

            fs /= len( dataIter.dataset )
            fs_grad /= len( dataIter.dataset )
            fs_hessian /= len( dataIter.dataset )
               
            # now compute the error terms here for the entire dataset
            first_error = torch.abs(( fr  - ( fs + fs_grad )) / fr)
            second_error = torch.abs(( fr - ( fs + fs_grad + fs_hessian)) / fr)

            print( fr, fs.cpu().data[0], fs_grad.cpu().data[0], fs_hessian.cpu().data[0] )
            dx.append( torch.norm( flat_random ).data[0] )
            dx_2.append( first_error.cpu().data[0] )
            dx_3.append( second_error.cpu().data[0] )

            c = [ 0.5 * W for W in c ]
            
        #return the results here
        return dx, dx_2, dx_3

