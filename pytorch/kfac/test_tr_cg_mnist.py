__author__ = 'Sudhir Kylasa'

import os
import datetime
import sys 
import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from torchvision.transforms import transforms

from mnist_loader import get_mnist_data_loader 

from lenet_mnist import LeNetMNIST

from conjugate_gradient import ConjugateGradient
from subsampled_tr_cg import SubsampledTRCG

from subsampled_tr_cg_bfgs import SubsampledTRCGBFGS
from BFGSUpdate import BFGSUpdate

from tr_cg_utils import SubsampledTRCGParams
from tr_cg_utils import SubsampledTRCGStatistics

print( "======== TR CG Solver =========")

#Path variables here. 
data_path = '/home/skylasa/datasets/raw-data/mnist/'

DEVICE = int( sys.argv[1] )
NETWORK = sys.argv[2]
ACTIVATION = sys.argv[3]
INITIALIZATION = sys.argv[4]
LOGFILE = sys.argv[5]
SCALE = float ( sys.argv[6] )

print( "...begin")

torch.manual_seed( 999 )
torch.cuda.manual_seed_all( 999 )

torch.cuda.set_device( DEVICE )

#Read the dataset here. 
batch_size=1024
train_size = 50000

sampler = WeightedRandomSampler( np.ones( train_size ) * (1./np.float(train_size)) , train_size, replacement=True )
cifar_train, cifar_sampler = get_mnist_data_loader( cvs_path=data_path, batch_size=batch_size,matFile='train_mat.txt', vecFile='train_vec.txt', sampler=sampler)
cifar_test, cifar_test_sampler = get_mnist_data_loader( cvs_path=data_path, batch_size=batch_size,matFile='test_mat.txt', vecFile='test_vec.txt', sampler=sampler)
	

print( "...done loading data")
print( len( cifar_train.dataset) )
print( len( cifar_test.dataset ) )
#exit ()

#model
model = LeNetMNIST( num_classes = 10, activation = ACTIVATION); 
model.double ()
model.cuda ()

#criterion = nn.MSELoss(size_average=False, reduce=True)
#criterion = nn.CrossEntropyLoss(size_average=False )
#model.setLossFunction( criterion )
print( "...done network creation")

#Subsampled Trust Region solver here. 
trparams = SubsampledTRCGParams ()
trparams.delta = np.float64( 1200 )
trparams.max_delta = np.float64( 12000 )
trparams.eta1 = np.float64 (0.8)
trparams.eta2 = np.float64 (1e-4)
trparams.gamma1 = np.float64 (2)
trparams.gamma2 = np.float64 (1.2)
trparams.max_props = float('Inf')
trparams.max_mvp = float('Inf')

trparams.max_iters = 500
trparams.sampleSize =  int( len(cifar_train.dataset) * 0.1 )

print( ".... sample Size: %d " % (trparams.sampleSize) )

trstats = SubsampledTRCGStatistics ( LOGFILE , console=True )
trstats.printHeader ()
print( "...done initialization params")

bfgsupdate = BFGSUpdate( 50, model.getParamLen () )

model.initOffsets ()
#model.initRandomWeights ()
#trsolver = SubsampledTRCG( model, ConjugateGradient, trparams, trstats, cifar_train, cifar_test, cifar_sampler, INITIALIZATION, SCALE )
trsolver = SubsampledTRCGBFGS( model, ConjugateGradient, trparams, trstats, cifar_train, cifar_test, cifar_sampler, INITIALIZATION, SCALE, bfgsupdate )
print( "...created the solver")

#
# Initialize the starting point
#
#trsolver.initializeZero ()
#trsolver.initializeRandom ()
#trsolver.initKaiming ()
trsolver.initXavier ()


#
# and.... ACTION !!!
#
print("...begin solving TR")
print 
print
trsolver.solve ()
print
print("...done with solver")



#
# and... SOLVER !!!
#
trstats.shutdown ()
print
print("....ALL DONE....")
