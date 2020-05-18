
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

#global Zis
#global dxa

def CONVHOOK1( module, grad_input, grad_output ):  
   global dxa
   #print( grad_output[0].data.shape )
   dxa[ 0 ].copy_( grad_output[0].data )
def CONVHOOK2( module, grad_input, grad_output ):  
   global dxa
   dxa[ 1 ].copy_( grad_output[0].data )
def FCHOOK1( module, grad_input, grad_output ):  
   global dxa
   dxa[ 2 ].copy_( grad_output[0].data )
def FCHOOK2( module, grad_input, grad_output ):  
   global dxa
   dxa[ 3 ].copy_( grad_output[0].data )
def FCHOOK3( module, grad_input, grad_output ):  
   global dxa
   dxa[ 4 ].copy_( grad_output[0].data )

class LeNetCIFARKFAC(nn.Module):

    def __init__(self, num_classes=10, activation='relu', batchSize=1000, kfac=None, bias=False):
        super(LeNetCIFARKFAC, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias=bias)
        self.conv1.register_backward_hook( CONVHOOK1 )

        self.conv2 = nn.Conv2d(6, 16, 5, bias=bias)
        self.conv2.register_backward_hook( CONVHOOK2 )

        self.fc1   = nn.Linear(16*5*5, 120, bias=bias)
        self.fc1.register_backward_hook( FCHOOK1 )

        self.fc2   = nn.Linear(120, 84, bias=bias) 
        self.fc2.register_backward_hook( FCHOOK2 )

        self.fc3   = nn.Linear(84, 10, bias=bias) 
        self.fc3.register_backward_hook( FCHOOK3 )

        self.offsets = [0]
        self.activation = activation
        self.batchSize = batchSize
        self.KFACEnabled = False
        self.kfac = kfac
        self.bias = bias

        global Zis
        global dxa
        Zis = []
        dxa = []

    def startRecording( self ): 
        self.KFACEnabled = True

    def stopRecording( self ): 
        self.KFACEnabled = False

    def initKFACStorage( self ): 
        global Zis
        global dxa
        Zis.append( None )
        Zis.append( torch.zeros( (self.batchSize, 6, 14, 14) ).type(torch.cuda.DoubleTensor) )
        Zis.append( torch.zeros( (self.batchSize, 16, 5, 5) ).type(torch.cuda.DoubleTensor ) )
        Zis.append( torch.zeros( (self.batchSize, 120 )).type(torch.cuda.DoubleTensor ) )
        Zis.append( torch.zeros( (self.batchSize, 84 )).type(torch.cuda.DoubleTensor ) )

        #Fill the dxA's now...
        dxa.append( torch.zeros( (self.batchSize, 6, 28, 28) ).type(torch.cuda.DoubleTensor) )
        dxa.append( torch.zeros( (self.batchSize, 16, 10, 10 )).type(torch.cuda.DoubleTensor) )
        dxa.append( torch.zeros( (self.batchSize, 120)).type(torch.cuda.DoubleTensor ) )
        dxa.append( torch.zeros( (self.batchSize, 84)).type(torch.cuda.DoubleTensor ) )
        dxa.append( torch.zeros( (self.batchSize, 10)).type(torch.cuda.DoubleTensor ) )


    def forward(self, x): 
        if self.activation == 'softplus': 
            out = self.conv1( x )
            out = F.softplus(out)
            out = F.avg_pool2d(out, 2)
            if (self.KFACEnabled ): 
                Zis[ 1 ].copy_( out.data )

            out = self.conv2( out )
            out = F.softplus(out)
            out = F.avg_pool2d(out, 2)
            if (self.KFACEnabled): 
                Zis[ 2 ].copy_( out.data )

            out = out.view(out.size(0), -1) 

            out = self.fc1( out )
            out = F.softplus(out)
            if( self.KFACEnabled ): 
                Zis[ 3 ].copy_( out.data )

            out = self.fc2( out )
            out = F.softplus(out)
            if( self.KFACEnabled ): 
                Zis[ 4 ].copy_( out.data )

            out = self.fc3(out)

            return out
        elif self.activation == 'elu': 
            out = F.elu(self.conv1(x))
            out = F.avg_pool2d(out, 2)
            out = F.elu(self.conv2(out))
            out = F.avg_pool2d(out, 2)
            out = out.view(out.size(0), -1) 
            out = F.elu(self.fc1(out))
            out = F.elu(self.fc2(out))
            out = self.fc3(out)
            return out
        elif self.activation == 'relu': 
            out = F.relu(self.conv1(x))
            out = F.avg_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.avg_pool2d(out, 2)
            out = out.view(out.size(0), -1) 
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
            return out
        elif self.activation == 'tanh': 
            out = nn.Tanh(self.conv1(x))
            out = F.avg_pool2d(out, 2)
            out = nn.Tanh(self.conv2(out))
            out = F.avg_pool2d(out, 2)
            out = out.view(out.size(0), -1) 
            out = nn.Tanh(self.fc1(out))
            out = nn.Tanh(self.fc2(out))
            out = self.fc3(out)
            return out

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
        #return torch.cat( [ grad.contiguous ().view( -1 ) for grad in g if grad is not None ] )
        return g

    def computeGradientIter( self, X, Y, regularization): 

        self.zero_grad ()
        x_var = Variable( X )
        y_var = Variable( Y )

        loss, pred = self.evalModelChunk( x_var, y_var )
        accu = pred.eq( y_var ).type(torch.LongTensor).sum ()
        g = self._computeGradient( loss, False )

        x_var.volatile = True
        y_var.volatile = True

        #W = self.getWeights ()

        #lossreg = 0.5 * regularization * torch.dot( W, W )
        #gradreg = regularization * W

        #return (loss.cpu().data[0] / X.shape[0] + lossreg ), (g.data / X.shape[0] + gradreg), accu.data[0]
        return g
        
    def computeHv( self, sampleX, sampleY, vec, regularization ): 
        self.zero_grad ()
        loss, pred = self.evalModelChunk( sampleX, sampleY )
        gradient = self._computeGradient( loss, True )
        self.zero_grad ()
        hv = self._computeGradient( (Variable( vec ) * gradient).sum (), False )

        W = self.getWeights ()
        return hv / sampleX.shape[0] + regularization * Variable( vec )

    def computeHv2( self, sampleX, sampleY, vec, regularization ): 
        #compute using g(W +ev) - g(W) / e		
        loss1, pred = self.evalModelChunk( sampleX, sampleY )
        gw = self._computeGradient( loss1, False )

        self.updateWeights( 1e-4 * vec )
        loss2, pred = self.evalModelChunk( sampleX, sampleY )
        gwe = self._computeGradient( loss2, False )

        self.updateWeights( -1e-4 * vec )
        return (gwe - gw) / 1e-4

    def computeKFACHv( self, X, vec ): 
        global Zis
        global dxa

        self.kfac.updateZAndD( Zis, dxa )
        self.kfac.prepMatVec( X )
        return self.kfac.computeMatVec( vec )
        

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
        self.conv2.weight.data.copy_( c3 )
        self.fc1.weight.data.copy_( l1 )
        self.fc2.weight.data.copy_( l3 )
        self.fc3.weight.data.copy_( l5 )
 
        if (self.bias == True): 
            self.conv1.bias.data.copy_( c2 )
            self.conv2.bias.data.copy_( c4 )
            self.fc1.bias.data.copy_( l2 )
            self.fc2.bias.data.copy_( l4 )
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

