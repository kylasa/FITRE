
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import readWeights

import numpy as np


class Swish(nn.Module):
    """
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

#
#	Forward
#
m = Swish ()
forward_data = autograd.Variable( torch.randn( 2 ) )
print( "Input Data: ", forward_data ) 
forward_pass = m( forward_data )
print( "Forward Pass: ",  forward_pass )
print ()
print ()
print ()

#
#	Gradient
#
g = autograd.grad( forward_pass, m.parameters (), create_graph=False)

