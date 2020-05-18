
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Multinomial

import numpy as np

from utils import group_product

#global Zis
#global dxa

TYPE = torch.cuda.FloatTensor


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



def CONVHOOK1( module, grad_input, grad_output ):  
   global dxa

   #print("lol")
   #import pdb;pdb.set_trace();
   #print( grad_output[0].data.shape )
   dxa[ 0 ].copy_( grad_output[0].data )
   #print( torch.norm( grad_input[ 0 ].data ))
   #print( torch.norm( grad_input[ 1 ].data ))
def CONVHOOK2( module, grad_input, grad_output ):  
   global dxa
   dxa[ 1 ].copy_( grad_output[0].data )
   #print( torch.norm( grad_input[ 0 ].data ) )
   #print( torch.norm( grad_input[ 1 ].data ) )
   #print( torch.norm( grad_input[ 2 ].data ) )
def FCHOOK1( module, grad_input, grad_output ):  
   global dxa
   #print("GANG!!!!!!!!!!!!!!!!!!!!!!!!!!!")

   dxa[ 2 ].copy_( grad_output[0].data )
def FCHOOK2( module, grad_input, grad_output ):  
   global dxa
   dxa[ 3 ].copy_( grad_output[0].data )
def FCHOOK3( module, grad_input, grad_output ):  
   global dxa
   dxa[ 4 ].copy_( grad_output[0].data )

def SWISHBACK( module, grad_input, grad_output ): 
	print( 'SWISHBACK... ', torch.norm( grad_output[0].data[0, 0, :, :] ) )
	print( 'SWISHBACK... before image', torch.norm( grad_output[0].data[0, 0, :, :] ) )
	print( 'SWISHBACK... before image', grad_output[ 0 ].data[ 0, 0, :, : ] ) 
	print( 'SWISHBACK... after image', torch.norm( grad_input[0].data[0, 0, :, :] ) )
	print( 'SWISHBACK... after image', grad_input[ 0 ].data[ 0, 0, :, : ] ) 

def POOLBACK( module, grad_input, grad_output ): 
	print( 'POOLBACK...', torch.norm( grad_output[ 0 ].data[0, 0, :, :] ) )
	print( 'POOLBACK...complete ', torch.norm( grad_output[ 0 ].data ) )
	print( 'POOLBACK...Img input to pooling', grad_output[ 0 ].data[ 0, 0, :, : ] )
	print( 'POOLBACK...Img output to pooling', grad_input[ 0 ].data[ 0, 0, :, : ] )

class ALEXNETCIFAR(nn.Module):

    def __init__(self, num_classes=10, activation='relu', batchSize=1000, kfac=None, bias=False,r = 1e-6, momentum = 0, check_grad = False):
        super(ALEXNETCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=bias)
        self.swish1=Swish()
        self.pool1 = nn.MaxPool2d( kernel_size=3, stride=2, padding=1)
        self.conv1.register_backward_hook( CONVHOOK1 )

        #self.pool1 = nn.AvgPool2d( kernel_size=2 )
        #self.swish1.register_backward_hook( SWISHBACK )
        #self.pool1.register_backward_hook( POOLBACK )
        #self.conv1_bn = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride =1, padding=2, bias=bias)
        self.swish2=Swish ()
        self.conv2.register_backward_hook( CONVHOOK2 )
        self.pool2 = nn.MaxPool2d( kernel_size=3, stride=2, padding=1 )

        #self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1   = nn.Linear(64*8*8, 384, bias=bias)
        self.swish3=Swish ()
        self.fc1.register_backward_hook( FCHOOK1 )
        #self.l1_bn = nn.BatchNorm2d(120)
        self.fc2   = nn.Linear(384,192, bias=bias)
        self.swish4=Swish ()
        self.fc2.register_backward_hook( FCHOOK2 )
        #self.l2_bn = nn.BatchNorm2d(84) 
        self.fc3   = nn.Linear(192, num_classes, bias=bias)
        self.fc3.register_backward_hook( FCHOOK3 )
        #self.fc4   = 
        self.swish5 = Swish ()

        self.offsets = [0]
        self.activation = activation
        self.batchSize = batchSize
        self.KFACEnabled = False
        self.kfac = kfac
        self.bias = bias
        self.regularization = r
        self.momentum = momentum
        self.check_grad = check_grad
        global Zis
        global dxa
        Zis = []
        dxa = []

    def computeTempsForKFAC( self, X ):

        self.zero_grad ()
        x_var = Variable( X  )
        out = self (x_var)
	
        probs = F.softmax( out, dim=1 )
        #noise_outputs = Multinomial( 1, probs=probs).sample ().nonzero()[:,-1]
        noise_outputs = torch.max( probs, 1 )[1]
        loss = self.lossFunction( out, noise_outputs )
        #g = self._computeGradient2( loss, False )
        loss.backward(retain_graph=True )

    def unpackWeights( self, vec ):  
        idx = 0 
        w = []
        for z in self.parameters (): 
            x = torch.index_select( vec, 0, torch.arange( self.offsets[ idx ], self.offsets[ idx ] + z.numel () ).type( torch.cuda.LongTensor ) ).view( z.size () ) 
            w.append( x ) 

        return w

    def startRecording( self ): 
        self.KFACEnabled = True

    def stopRecording( self ): 
        self.KFACEnabled = False

    def initKFACStorage( self ): 
        global Zis
        global dxa
        Zis.append( None )
        Zis.append( torch.zeros( (self.batchSize, 64, 16, 16) ).type(TYPE) )
        Zis.append( torch.zeros( (self.batchSize, 64, 8, 8) ).type(TYPE) )
        Zis.append( torch.zeros( (self.batchSize, 384 )).type(TYPE) )
        Zis.append( torch.zeros( (self.batchSize, 192 )).type(TYPE) )

        #Fill the dxA's now...
        dxa.append( torch.zeros( (self.batchSize, 64, 32, 32) ).type(TYPE) )
        dxa.append( torch.zeros( (self.batchSize, 64, 16, 16 )).type(TYPE) )
        dxa.append( torch.zeros( (self.batchSize, 384)).type(TYPE) )
        dxa.append( torch.zeros( (self.batchSize, 192)).type(TYPE) )
        dxa.append( torch.zeros( (self.batchSize, 10)).type(TYPE) )


    def forward(self, x): 
        if self.activation == 'softplus': 
            #print(x.shape)
            out = self.conv1( x )
            #print( 'output of first conv: ', out[ 0, 0, :, : ] )
            #print( 'bias: ', self.conv1.bias.data )
            #print( 'Weights 1: ', self.conv1.weight.data[ 0, 0, :, : ] )
            #print( 'Data 1: ', x.data[ 0, 0, 0:5, 0:5 ] )
            #print( 'Weights 2: ', self.conv1.weight.data[ 0, 1, :, : ] )
            #print( 'Data 2: ', x.data[ 0, 1, 0:5, 0:5 ] )
            #print( 'Weights 3: ', self.conv1.weight.data[ 0, 2, :, : ] )
            #print( 'Data 3: ', x.data[ 0, 2, 0:5, 0:5 ] )

            out = self.swish1(out)
            

            #import pdb;pdb.set_trace();
            #out = F.max_pool2d(out,3, stride=2, padding =1 )
            #out = F.avg_pool2d(out,2)
            out = self.pool1( out )
            
            #print(out.shape)
            if (self.KFACEnabled ):
                #print(Zis[1].size(), out.data.size ())
                Zis[ 1 ].copy_( out.data )

            out = self.conv2( out )
            #print(out.shape)
            out = self.swish2(out)
            #out = F.max_pool2d(out,3, stride=2, padding = 1)
            #out = F.avg_pool2d(out,2)
            out = self.pool2( out )
            
            if (self.KFACEnabled): 
                Zis[ 2 ].copy_( out.data )

            out = out.view(out.size(0), -1) 
            
            out = self.fc1( out )
            out = self.swish3(out)
            #print(out.shape)
            #out = Swish(out)
            
            if( self.KFACEnabled ): 
                Zis[ 3 ].copy_( out.data )

            out = self.fc2( out )
            #print(out.shape)
            out = self.swish4(out)
            #print(out.shape)
            if( self.KFACEnabled ): 
                Zis[ 4 ].copy_( out.data )

            out = self.fc3(out)
            #out = self.swish5( out )
            


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

    def _computeGradient2( self, func, create_graph): 

        '''
        func.backward ()
        g = []
        for p in self.parameters (): 
            g.append( p.grad.data )
        return g
        '''

        if create_graph: 
            g = autograd.grad( func, self.parameters (), create_graph=True )
        else:
            g = autograd.grad( func, self.parameters (), create_graph=False )
        return g

    def computeGradientIter2( self, x_var, y_var ): 

        self.zero_grad ()
        outputs = self( x_var )
        loss = self.lossFunction( outputs, y_var )
        #g = self._computeGradient2( loss, True)
        loss.backward( create_graph=True )

        #Add regularization
        g = []
        for p in  self.parameters () :
            t = p.grad.data.clone ()
            t.add_( self.regularization, p.data )
            g.append( t )
        return g

    def computeHv_r( self, sampleX, sampleY, vec ): 
        self.zero_grad ()
        loss, pred = self.evalModelChunk( sampleX, sampleY )
        gradient = self._computeGradient2( loss, True )

        #print( 'computeHV_r: gradient: ', group_product( gradient, gradient ) )
        #self.zero_grad ()
        #print( 'computeHV_r: gradient: ', group_product( gradient, gradient ) )

        Hv = autograd.grad( gradient, self.parameters (), grad_outputs=vec, only_inputs=True, retain_graph=True )
        Hv = [ hd.detach () + self.regularization * d for hd, d in zip(Hv, vec) ]
        return Hv

    def computeKFACHv( self, X, vec ): 
        global Zis
        global dxa

        self.kfac.updateZAndD( Zis, dxa )
        self.kfac.prepMatVec( X ) 
        return self.kfac.computeMatVec( vec )
        

    def evalModel( self, x_var, y_var): 
        value = 0
        ll = 0
        accu = 0
        count=0

        torch.cuda.empty_cache()
        out = self( x_var )
        cur_loss = self.lossFunction( out, y_var )

        value += cur_loss
        ll += cur_loss
        accu += (torch.max( out, 1)[1].eq( y_var ) ).cpu ().type(torch.LongTensor).sum ().item ()

        return ll.cpu ().item (),  np.float( accu )  * 100. / np.float( x_var.shape[0] )


    def evalModelChunk( self, x, y ): 
        #print("!!")
        out = self (x)
       # print("!?")
        loss = self.lossFunction( out, y )
        #print("??")
        return loss, torch.max( out, 1)[1]

    def initWeights( self, val ): 
        for W in self.parameters (): 
            W.data.fill_(val)

    def setWeights( self, vec ): 
        idx = 0
        for W in self.parameters ():
           W.data.copy_( torch.index_select( vec, 0, torch.arange( self.offsets[idx], self.offsets[ idx ] + W.numel () ).type( torch.cuda.LongTensor ) ).view( W.size () ) )
           idx += 1

    def setWeightsAndBiases( self, c1, b1, c2, b2, l1, lb1, l2, lb2, l3, lb3 ): 
        self.conv1.weight.data.copy_( c1 )
        self.conv1.bias.data.copy_( b1 )
        self.conv2.weight.data.copy_( c2 )
        self.conv2.bias.data.copy_( b2 )

        self.fc1.weight.data.copy_( l1 )
        self.fc1.bias.data.copy_( lb1 )
        self.fc2.weight.data.copy_( l2 )
        self.fc2.bias.data.copy_( lb2 )
        self.fc3.weight.data.copy_( l3 )
        self.fc3.bias.data.copy_( lb3 )

    def setWeights( self, c1, c2, l1, l2, l3 ): 
        self.conv1.weight.data.copy_( c1 )
        self.conv2.weight.data.copy_( c2 )

        self.fc1.weight.data.copy_( l1 )
        self.fc2.weight.data.copy_( l2 )
        self.fc3.weight.data.copy_( l3 )


    def updateWeights( self, vec ): 
        idx = 0
        for W in self.parameters ():
           W.data.add_( torch.index_select( vec, 0, torch.arange( self.offsets[idx], self.offsets[ idx ] + W.numel () ).type( torch.cuda.LongTensor ) ).view( W.size () ) )
           idx += 1

    def initHybrid( self ): 
        self.initFromPeng( ) 


    def initFromPeng (self ): 
        for m in self.parameters ():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: 
                    nn.init.constant(m.bias, 0.0)

    def initXavierUniform( self ): 
        for W in self.parameters (): 
            if len(W.data.size () ) > 1: 
                nn.init.xavier_uniform( W.data ) 
            else: 
                W.data.random_(0, 4)
                W.data *= 0.1

    def initKaimingUniform(net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)

    def initconstant( net ): 
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_( 1 )				
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
                    m.bias.data.fill_( 1 )
            elif isinstance(m, nn.Linear):
                m.weight.data.fill_( 1 )
                if m.bias is not None:
                    m.bias.data.fill_( 1 )
