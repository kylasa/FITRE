
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

class AlexNetCIFAR(nn.Module):

    def __init__(self, num_classes=10, activation='relu'):
        super(AlexNetCIFAR, self).__init__()

        if activation == 'relu': 
            self.features = nn.Sequential( 
                nn.Conv2d( 3, 64, kernel_size=11, stride=4, padding=5), 
                nn.ReLU( inplace=True ),  
                nn.MaxPool2d( kernel_size=2, stride=2), 

                nn.Conv2d( 64, 192, kernel_size=5, padding=2), 
                nn.ReLU( inplace=True ),  
                nn.MaxPool2d( kernel_size=2, stride=2), 

                nn.Conv2d( 192, 384, kernel_size=3, padding=1), 
                nn.ReLU( inplace=True ),  

                nn.Conv2d( 384, 256, kernel_size=3, padding=1), 
                nn.ReLU( inplace=True ),  

                nn.Conv2d( 256, 256, kernel_size=3, padding=1), 
                nn.ReLU( inplace=True ),  
                nn.MaxPool2d( kernel_size=2, stride=2), 
            )   
        elif activation == 'softplus': 
            self.features = nn.Sequential( 
                nn.Conv2d( 3, 64, kernel_size=11, stride=4, padding=5), 
                nn.Softplus( ),  
                nn.MaxPool2d( kernel_size=2, stride=2), 

                nn.Conv2d( 64, 192, kernel_size=5, padding=2), 
                nn.Softplus( ),  
                nn.MaxPool2d( kernel_size=2, stride=2), 

                nn.Conv2d( 192, 384, kernel_size=3, padding=1), 
                nn.Softplus( ),  

                nn.Conv2d( 384, 256, kernel_size=3, padding=1), 
                nn.Softplus( ),  

                nn.Conv2d( 256, 256, kernel_size=3, padding=1), 
                nn.Softplus( ),  
                nn.MaxPool2d( kernel_size=2, stride=2), 
            )   
        else: 
            self.features = nn.Sequential( 
                nn.Conv2d( 3, 64, kernel_size=11, stride=4, padding=5), 
                nn.ELU( inplace=True ),  
                nn.MaxPool2d( kernel_size=2, stride=2), 

                nn.Conv2d( 64, 192, kernel_size=5, padding=2), 
                nn.ELU( inplace=True ),  
                nn.MaxPool2d( kernel_size=2, stride=2), 

                nn.Conv2d( 192, 384, kernel_size=3, padding=1), 
                nn.ELU( inplace=True ),  

                nn.Conv2d( 384, 256, kernel_size=3, padding=1), 
                nn.ELU( inplace=True ),  

                nn.Conv2d( 256, 256, kernel_size=3, padding=1), 
                nn.ELU( inplace=True ),  
                nn.MaxPool2d( kernel_size=2, stride=2), 
            )   
        self.classifier = nn.Linear( 256, num_classes )
        self.offsets = [0]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def initOffsets (self): 
        for W in self.parameters (): 
            self.offsets.append( W.numel () )
        self.offsets = np.cumsum( self.offsets )

    def setLossFunction( self, ll ): 
        self.lossFunction = ll

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

    def _computeGradient( self, func, create_graph): 
        if create_graph: 
            g = autograd.grad( func, self.parameters (), create_graph=True )
        else:
            g = autograd.grad( func, self.parameters () )
        return torch.cat( [ grad.contiguous ().view( -1 ) for grad in g ] )

    def _computeGradBackward( self, func ): 
        func.backward( )

    def computeGradientIter( self, data  ): 
        self.zero_grad ()
        #self.initWeights( pt )

        g = 0
        ll = 0
        cum_loss = 0
        accu = 0
        for X, Y in data: 
            x_var = Variable( X.type(torch.DoubleTensor ).cuda () )
            y_var = Variable( Y.cuda () )
            lll, loss, pred = self.evalModelChunk( x_var, y_var )
            ll += lll
            cum_loss += loss
            #accu += (pred == torch.max( Y, 1)[1].numpy ()).sum ()
            accu += (pred.eq( y_var )).cpu ().type( torch.LongTensor).sum ().data[0]

            g += self._computeGradient( loss, False )
            x_var.volatile = True
            y_var.volatile = True

        #return (ll.cpu().data[0] / len( data.dataset ) ), (cum_loss.cpu ().data[0] / len( data.dataset )), ( g.data / len( data.dataset )), np.float(accu) * 100. / np.float(len( data.dataset ))
        return  (cum_loss.cpu ().data[0] / len( data.dataset )), ( g.data / len( data.dataset )), np.float(accu) * 100. / np.float(len( data.dataset ))
        
    def computeHv( self, sampleX, sampleY, vec ): 
        self.zero_grad ()
        #self.initWeights( evalPt )
        hv = 0

        ll, loss, pred = self.evalModelChunk( sampleX.cuda (), sampleY.cuda () )
        gradient = self._computeGradient( loss, True )
        hv = self._computeGradient( (Variable( vec ) * gradient).sum (), False )
        
        return hv / len( sampleY )

    def evalModel( self, dataIter ): 
        value = 0
        ll = 0
        accu = 0

        for data, label in dataIter: 
            x_var = Variable( data.type( torch.DoubleTensor ).cuda () )
            y_var = Variable( label.cuda () )
            out = self( x_var )
            cur_loss = self.lossFunction( out, y_var )
            value += cur_loss
            ll += cur_loss
            #accu += (torch.max( out, 1)[1].eq( y_var ) ).cpu ().sum ().data[0]
            accu += (torch.max( out, 1)[1].eq( y_var ) ).cpu ().type(torch.LongTensor).sum ().data[0]

            x_var.volatile=True
            y_var.volatile=True

        #return (ll.cpu ().data[0] / len( dataIter.dataset) ),  (value.cpu().data[0] / len( dataIter.dataset )), np.float( accu )  * 100. / np.float( len( dataIter.dataset ) )
        return (value.cpu().data[0] / len( dataIter.dataset )), np.float( accu )  * 100. / np.float( len( dataIter.dataset ) )

    def evalModelChunk( self, x, y ): 
        out = self (x)
        loss = self.lossFunction( out, y )
        return loss, loss, torch.max( out, 1)[1]

    def runDerivativeTest( self, numPoints, stPoint, randPoint, dataIter ): 

        fr = 0
        c = [ Variable(W, requires_grad=True).type ( torch.DoubleTensor) for W in randPoint ]
        self.initWeights( c )
        ll, fr = self.evalModel( dataIter, False)

        dx = [] #norm of the noise
        dx_2 = [] # diff squared
        dx_3 = [] #diff cubed

        for i in range( numPoints ): 
            
            fs = 0    
            fs_grad = 0
            fs_hessian = 0
            flat_random = torch.cat( [ p.contiguous().view( -1 ) for p in c ] )
            idx = 0

            for X, Y in dataIter: 

                x_var = Variable( X )
                y_var = Variable( Y )

                self.zero_grad ()
                self.initWeights( stPoint )
                ll, fs_loss = self.evalModelChunk( x_var, y_var )
                gs = self.computeGradient( fs_loss, True )
                hv = self.computeGradient( (flat_random * gs).sum (), False )

                x_var.volatile = True
                y_var.volatile = True

                first_order = torch.dot( flat_random, gs )
                second_order = 0.5 * torch.dot( flat_random, hv )

                fs += fs_loss 
                fs_grad += first_order
                fs_hessian += second_order
               
            # now compute the error terms here for the entire dataset
            first_error = torch.abs( fr  - ( fs + fs_grad )) / fs
            second_error = torch.abs( fr - ( fs + fs_grad + fs_hessian)) / fs

            print( fr.data[0], fs.data[0], fs_grad.data[0], fs_hessian.data[0] )
            dx.append( torch.norm( flat_random ).data )
            dx_2.append( first_error.data )
            dx_3.append( second_error.data )

            c = [ 0.5 * W for W in c ]
            
        #return the results here
        return dx, dx_2, dx_3

