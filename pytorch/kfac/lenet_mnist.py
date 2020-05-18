
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

class LeNetMNIST(nn.Module):

    def __init__(self, num_classes=10, activation='relu'):
        super(LeNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1   = nn.Linear(320, 50)
        self.fc2   = nn.Linear(50, 10)
        self.offsets = [0]
        self.activation = activation

    def forward(self, x):
        if self.activation == 'softplus':
            out = F.avg_pool2d( F.softplus( self.conv1( x ) ), 2 )
            out = F.avg_pool2d( F.softplus( self.conv2( out )), 2 )
            out = out.view( -1, 320 )
            out = F.softplus( self.fc1( out ) )
            out = self.fc2( out )
            out = F.log_softmax( out, dim=1 )
            return out
        elif self.activation == 'elu':
            out = F.avg_pool2d( F.elu( self.conv1( x )),2 )
            out = F.avg_pool2d( F.elu( self.conv2( out )),2 )
            out = out.view( -1, 320 )
            out = F.elu( self.fc1( out ) )
            out = self.fc2( out )
            out = F.log_softmax( out, dim=1 )
            return out
        elif self.activation == 'relu':
            out = F.avg_pool2d( F.relu( self.conv1( x ) ),2 )
            out = F.avg_pool2d( F.relu( self.conv2( out )),2 )
            out = out.view( -1, 320 )
            out = F.relu( self.fc1( out ) )
            out = self.fc2( out )
            out = F.log_softmax( out, dim=1 )
            return out


    def initOffsets (self): 
        for W in self.parameters (): 
            self.offsets.append( W.numel () )
        self.offsets = np.cumsum( self.offsets )


    def getParamLen( self ):
        l = 0
        for W in self.parameters ():
            l += W.numel ()
        return l

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
	tc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	matched = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for X, Y in data: 

            self.zero_grad ()
            x_var = Variable( X.type( torch.DoubleTensor ).cuda () )
            t = torch.squeeze( Y, dim=1 ) 
            y_var = Variable( t.type( torch.LongTensor ).cuda () )

            loss, pred = self.evalModelChunk( x_var, y_var )
            cum_loss += loss
            accu += (pred.eq( y_var )).cpu ().type (torch.LongTensor).sum ().data[0]

            '''
            l = pred.cpu().data
            for m in l: 
                counters[ m ] += 1

            for m in Y.numpy (): 
                tc[ m[0] ] += 1

            for a, b in zip( l, Y.numpy () ):
                if a == b[0]: 
                   matched[ a ] += 1

            #print( pred )
            #print ( y_var.cpu () )
            #print ( pred.eq( y_var ) )
            #print( accu )
            #print( sum( matched ) )
            #print
            '''

            g += self._computeGradient( loss, False )
            x_var.volatile = True
            y_var.volatile = True

        print (accu)
        #print( counters )
        #print( tc )
        return (cum_loss.cpu ().data[0] / len( data.dataset )), (g.data / len( data.dataset )), np.float(accu) * 100. / np.float(len( data.dataset ))
        
    def computeHv( self, sampleX, sampleY, vec ): 
        loss, pred = self.evalModelChunk( sampleX, torch.squeeze(sampleY, dim=1) )
        gradient = self._computeGradient( loss, True )
        hv = self._computeGradient( (Variable( vec, requires_grad=False ) * gradient).sum (), False )
        return hv / len( sampleY )

    def computeGv( self, sampleX, sampleY, vec ): 
        #forward pass to to compute Jv. 
        loss, pred = self.evalModelChunk( sampleX, torch.squeeze( sampleY, dim=1) )
        gradient = self._computeGradient( loss, True )

        varVec = Variable( vec, requires_grad=False)

        Jv = gradient * varVec

        #hessian vector product here. 
        HJv = self._computeGradient( (Jv * gradient).sum (), True )

        #Back Prop with the vector as initial vector
        self.zero_grad () 
        JHJv = self._computeGradient( (varVec * gradient).sum (), False )

        return JHJv / len( sampleY )

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

    def getDerivatives( self ): 
        return torch.cat([ w.grad.contiguous().view( -1 ).data.clone () for w in self.parameters () ])

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
            l = torch.squeeze( label, dim=1 )
            y_var = Variable( l.type( torch.LongTensor ).cuda () )
            out = self( x_var )
            #cur_loss = self.lossFunction( out, y_var )
            cur_loss = F.nll_loss( out, y_var, size_average=False )
            value += cur_loss

            accu += (torch.max( out, 1)[1].eq( y_var ) ).cpu ().type(torch.LongTensor).sum ().data[0]

            x_var.volatile=True
            y_var.volatile=True

        print( accu )
        return (value.cpu().data[0] / len( dataIter.dataset )), np.float( accu )  * 100. / np.float( len( dataIter.dataset ) )

    def evalModelChunk( self, x, y ): 
        out = self (x)
        #loss = self.lossFunction( out, y )
        loss = F.nll_loss( out, y, size_average=False )
        return loss, torch.max( out, 1)[1]

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

