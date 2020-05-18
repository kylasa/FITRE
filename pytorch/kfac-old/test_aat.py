__author__ = 'Sudhir Kylasa'

import os
import datetime
import sys
import numpy as np
import math

import torch
from torch import nn
from torch.autograd import Variable

from dataset_loader import TensorDataset
from lenet_cifar_kfac import LeNetCIFARKFAC
from kfac_utils import KFACUtils

#torch.cuda.set_device( 1 )

#Read the dataset here. 
data_path = '/scratch/skylasa/solvers/cnn-data/cifar-10-batches-py/'
train = TensorDataset( csv_path=data_path, fileFilter='data_batch_1' )
train_data, train_labels = train.getDataset ()

train_data = torch.from_numpy( train_data ).type(torch.cuda.DoubleTensor)
train_labels = torch.from_numpy( train_labels ).type(torch.cuda.LongTensor)

BATCH_SIZE = 1000


#compute a a^t and test whether it is invertible or not.

A = train_data[0:BATCH_SIZE, :, :, : ]

A_exp = A.unfold( 2, 5, 1).unfold( 3, 5, 1)
A_exp = A_exp.permute( 0, 2, 3, 1, 4, 5)
s = A_exp.size ()
A_exp = A_exp.contiguous ().view( s[0] * s[1] * s[2], s[3] * s[4] * s[5] )
A_exp = torch.cat( [ A_exp, torch.ones( s[0] * s[1] * s[2], 1).type( torch.cuda.DoubleTensor ) ], dim=1 )

print
print 
print( 'A matrix dimentions are: ', A_exp.size () )


#
# A_exp  = ( 1000 * 28 * 28 X 3 * 5 * 5 )
#

exp_A = torch.mm( A_exp.permute( 1, 0 ), A_exp )
exp_A = exp_A / BATCH_SIZE

print
print
print
print( ' A A^T is as follows: ....', exp_A.size () )
print
print
print

# compute the inverse of this matrix here. 

inv_exp_A = torch.inverse( exp_A )

print
print
print
print
print( 'Inverse computation is done.... ' )
