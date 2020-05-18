
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

TYPE = torch.cuda.DoubleTensor


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

class ALEXNETCIFAR(nn.Module):

    def __init__(self, num_classes=10, bias=False ):

        super(ALEXNETCIFAR, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=bias)
        self.swish1=Swish()
        self.pool1 = nn.MaxPool2d( kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride =1, padding=2, bias=bias)
        self.swish2=Swish ()
        self.pool2 = nn.MaxPool2d( kernel_size=3, stride=2, padding=1 )

        self.fc1   = nn.Linear(64*8*8, 384, bias=bias)
        self.swish3=Swish ()

        self.fc2   = nn.Linear(384,192, bias=bias)
        self.swish4=Swish ()

        self.fc3   = nn.Linear(192, num_classes, bias=bias)


    def forward(self, x): 
            out = self.conv1( x )
            out = self.swish1(out)
            out = self.pool1( out )
            
            out = self.conv2( out )
            out = self.swish2(out)
            out = self.pool2( out )
            
            out = out.view(out.size(0), -1) 
            
            out = self.fc1( out )
            out = self.swish3(out)
            
            out = self.fc2( out )
            out = self.swish4(out)

            out = self.fc3(out)
            return out

    def setLossFunction( self, ll ): 
        self.lossFunction = ll

    def computeGradientIter2( self, x_var, y_var ): 

        self.zero_grad ()
        outputs = self( x_var )
        loss = self.lossFunction( outputs, y_var )
        print( 'Loss: ', loss )

        loss.backward( create_graph=True )

        #Add regularization
        g = []
        for p in  self.parameters () :
            t = p.grad.data.clone ()
            g.append( t )
        return g

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
        out = self (x)
        loss = self.lossFunction( out, y )
        return loss, torch.max( out, 1)[1]

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
