__author__ = 'Sudhir Kylasa'

import os
import datetime
import sys 
import math
import numpy as np

import readWeights

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from torchvision import datasets, transforms


#from torchvision.transforms import transforms

from dataset_loader import get_tensor_data_loader 
from dataset_loader import get_cifar_data_loader
from dataset_loader_normalized import getNormalizedCIFAR
from dataset_loader_normalized import get_cifar_large_sampler_normalized
from alexnet_cifar import ALEXNETCIFAR
from lenet_cifar import LeNetCIFAR
from lenet_cifar_kfac import LeNetCIFARKFAC
from kfac_utils import KFACUtils

from conjugate_gradient import ConjugateGradient
from subsampled_tr_cg import SubsampledTRCG
from subsampled_tr_cg_kfac import SubsampledTRCGKFAC
from subsampled_tr_cg_kfac_full import SubsampledTRCGKFACFULL
from subsampled_tr_cg_kfac_frank_wolfe import SubsampledTRCGKFACFrankWolfe
from tr_cg_utils import SubsampledTRCGParams
from tr_cg_utils import SubsampledTRCGStatistics

print( "======== TR CG Solver =========")

#Path variables here. 
train_data = '/scratch/skylasa/solvers/cnn-data/'
test_data = '/scratch/skylasa/solvers/cnn-data/'
data_path = '/scratch/skylasa/solvers/cnn-data/'


DEVICE = 0
ACTIVATION = 'softplus'
LOGFILE = './test_noda.txt'

print( "...begin")

torch.manual_seed( 1 )
#torch.cuda.manual_seed_all( 1 )
torch.cuda.manual_seed( 1 )
torch.cuda.set_device( DEVICE )

#Read the dataset here. 
batch_size=500

torch.set_printoptions(precision=10)


'''
transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
'''
'''
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])  
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])  
'''

transform_train = transforms.Compose([
        transforms.ToTensor()
    ])  
transform_test = transforms.Compose([
        transforms.ToTensor()
    ])  


cifar_train, cifar_sampler = get_cifar_data_loader( csv_path = train_data, batch_size=batch_size, transform=transform_train)
cifar_train_test, cifar_sampler = get_cifar_data_loader( csv_path = train_data, batch_size=batch_size, train=True, transform=transform_test)
cifar_test, cifar_test_sampler = get_cifar_data_loader( csv_path=test_data, batch_size=batch_size, train=False, transform=transform_test)

'''
cifar_train, cifar_sampler = get_tensor_data_loader( cvs_path=train_data, batch_size=batch_size, shuffle=False )
cifar_train_test, cifar_sampler = get_tensor_data_loader( cvs_path=train_data, batch_size=batch_size, shuffle=False )
cifar_test, cifar_test_sampler = get_tensor_data_loader( cvs_path=test_data, fileFilter='test_batch*', batch_size=batch_size, shuffle=False)
'''

#cifar_train, cifar_sampler, cifar_test = getNormalizedCIFAR( cvs_path=data_path, batch_size=batch_size, shuffle=False, sampler=sampler )
#cifar_large_sampler = get_cifar_large_sampler_normalized( cifar_train, large_batch_size, sampler )

	
print( "...done loading data")
print( len( cifar_train.dataset) )
print( len( cifar_test.dataset ) )

print
print
print

#kfac here. 
bias=True

damp_gamma  	= 1e-2		#Levenberg-Marquardt Dampening Term 
regLambda 		= 0  		# Regularization Term
stats_decay		= 0.8			# stats decay term for Natural Gradient

momentum 		= 0 			#moving average -- momemtum
check_grad		= False # if need to run KFAC + Grad


'''
kfac  = KFACUtils( ksize = 5, padding = 0, layers = ['conv', 'conv', 'lin', 'lin', 'lin' ], batchSize = batch_size, bias=bias, debug=False, gamma = damp_gamma, regLambda = regLambda, stats_decay = stats_decay)

model = LeNetCIFARKFAC( num_classes = 10, activation = ACTIVATION, batchSize=batch_size, kfac=kfac, bias=bias, r=regLambda); 
model.double ()
model.cuda ()
'''

kfac  = KFACUtils( ksize = 5, padding = 2, layers = ['conv', 'conv', 'lin', 'lin', 'lin' ], batchSize = batch_size, bias=bias, debug=False, gamma = damp_gamma, regLambda = regLambda, stats_decay = stats_decay)

model = ALEXNETCIFAR( num_classes = 10, activation = ACTIVATION, batchSize=batch_size, kfac=kfac, bias=bias,r=regLambda, momentum = momentum, check_grad = check_grad ); 
#model.double ()
model.cuda ()

#criterion = nn.MSELoss(size_average=False)
#criterion = nn.MSELoss(size_average=False, reduce=True)
#criterion = nn.CrossEntropyLoss( )
criterion = nn.CrossEntropyLoss()
model.setLossFunction( criterion )
print( "...done network creation")

#Subsampled Trust Region solver here. 
trparams = SubsampledTRCGParams ()
trparams.delta = np.float64( 1. )
trparams.max_delta = np.float64( 100 )
trparams.min_delta = np.float64( 1e-6 )
trparams.eta1 = np.float64 (0.8)
trparams.eta2 = np.float64 (1e-4)
trparams.gamma1 = np.float64 (2)
trparams.gamma2 = np.float64 (1.2)
trparams.max_props = float('Inf')
trparams.max_mvp = float('Inf')

trparams.max_iters = 99
trparams.sampleSize =  int( len(cifar_train.dataset) * 0.1 )

print( ".... sample Size: %d " % (trparams.sampleSize) )

trstats = SubsampledTRCGStatistics ( LOGFILE , console=True )
trstats.printHeader ()
print( "...done initialization params")

model.initOffsets ()
#model.initRandomWeights ()
trsolver = SubsampledTRCGKFACFULL( model, ConjugateGradient, trparams, trstats, cifar_train, cifar_test, cifar_train_test, cifar_sampler, momentum, check_grad, False)
print( "...created the solver")

#
# Initialize the starting point
#
#trsolver.initializeZero ()
#trsolver.initializeRandom ()
#trsolver.initKaiming ()
#trsolver.initXavier ()
trsolver.initFromPeng ()
#trsolver.initConstant ()

#trsolver.initHybrid ()

exit ()




#
# and.... ACTION !!!
#
print("...begin solving TR")
print 
print

'''
temp = readWeights.readMatrix( './alexnet_xavier.txt', [ [64, 3, 5, 5], [64], [64, 64, 5, 5], [64], [384, 4096], [384], [192, 384], [192], [10, 192], [10] ] ); 
model.setWeights( torch.from_numpy( temp[0] ).type( torch.DoubleTensor ), torch.from_numpy(temp[1]).type( torch.DoubleTensor ), torch.from_numpy(temp[2]).type( torch.DoubleTensor ), torch.from_numpy(temp[3]).type( torch.DoubleTensor ), torch.from_numpy(temp[4]).type( torch.DoubleTensor ), torch.from_numpy(temp[5]).type( torch.DoubleTensor ), torch.from_numpy(temp[6]).type( torch.DoubleTensor ), torch.from_numpy(temp[7]).type( torch.DoubleTensor ), torch.from_numpy(temp[8]).type( torch.DoubleTensor ), torch.from_numpy(temp[9]).type( torch.DoubleTensor ) )
'''

'''
odict_keys(['features.0.weight', 'features.3.weight', 'classifier.0.weight', 'classifier.2.weight', 'classifier.4.weight'])
print( p.keys () )
'''

#import pdb;pdb.set_trace();

'''
p =  torch.load( '../peng/params.file' ) 
model.setWeights( p[ 'features.0.weight' ], p[ 'features.3.weight'], p[ 'classifier.0.weight' ], p[ 'classifier.2.weight' ], p['classifier.4.weight' ] )
'''

'''
odict_keys(['features.0.weight', 'features.3.weight', 'classifier.0.weight', 'classifier.2.weight', 'classifier.4.weight'])
'''

#model.load_state_dict( torch.load( '/scratch/skylasa/peng/params_test.file' ))
p =  torch.load( '/scratch/skylasa/peng/params_test.file' )
model.setWeights( p[ 'features.0.weight' ], p[ 'features.3.weight'], p[ 'classifier.0.weight' ], p[ 'classifier.2.weight' ], p['classifier.4.weight' ] )
print( 'Done loading the parameter file.... ' )

trsolver.solve ()
print
print("...done with solver")



#
# and... SOLVER !!!
#
trstats.shutdown ()
print
print("....ALL DONE....")
