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

from dataset_loader import get_tensor_data_loader 
from dataset_loader import get_cifar_data_loader
from dataset_loader_normalized import getNormalizedCIFAR
from dataset_loader_normalized import get_cifar_large_sampler_normalized
from alexnet_cifar import AlexNetCIFAR
from lenet_cifar import LeNetCIFAR
from lenet_cifar_kfac import LeNetCIFARKFAC
from kfac_utils import KFACUtils

from conjugate_gradient import ConjugateGradient
from subsampled_tr_cg import SubsampledTRCG
from subsampled_tr_cg_kfac import SubsampledTRCGKFAC
from subsampled_tr_cg_kfac_full import SubsampledTRCGKFACFULL
from subsampled_tr_cg_kfac_frank_wolfe import SubsampledTRCGKFACFrank
from tr_cg_utils import SubsampledTRCGParams
from tr_cg_utils import SubsampledTRCGStatistics

print( "======== TR CG Solver =========")

#Path variables here. 
train_data = '/scratch/skylasa/solvers/cnn-data/cifar-10-batches-py/'
test_data = '/scratch/skylasa/solvers/cnn-data/cifar-10-batches-py/'
data_path = '/scratch/skylasa/solvers/cnn-data/cifar-10-batches-py/'


DEVICE = int( sys.argv[1] )
NETWORK = sys.argv[2]
ACTIVATION = sys.argv[3]
LOGFILE = sys.argv[4]
TRANSFORMS = sys.argv[5]

print( "...begin")

torch.manual_seed( 999 )
torch.cuda.manual_seed_all( 999 )
torch.cuda.set_device( DEVICE )

#Read the dataset here. 
batch_size=256
train_size = 50000

if TRANSFORMS == 'no': 
	sampler = WeightedRandomSampler( np.ones( train_size ) * (1./np.float(train_size)) , train_size, replacement=True )
	cifar_train, cifar_sampler = get_tensor_data_loader( cvs_path=train_data, batch_size=batch_size, shuffle=False, sampler=sampler)
	cifar_test, cifar_test_sampler = get_tensor_data_loader( cvs_path=test_data, fileFilter='test_batch*', batch_size=batch_size, shuffle=False)

	'''
	cifar_train, cifar_sampler, cifar_test = getNormalizedCIFAR( cvs_path=data_path, batch_size=batch_size, shuffle=False, sampler=sampler )
	#cifar_large_sampler = get_cifar_large_sampler_normalized( cifar_train, large_batch_size, sampler )
	'''


	

'''
if TRANSFORMS == 'yes': 
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

	batch_size=1024
	train_size = 50000
	sampler = WeightedRandomSampler( np.ones( train_size ) * (1./np.float(train_size)) , train_size, replacement=True )

	#cifar_train, cifar_sampler = get_cifar_data_loader( csv_path = data_path, batch_size=batch_size, shuffle=True, transform=transform_train, sampler=sampler)
	#cifar_test, tt  = get_cifar_data_loader( csv_path = data_path, batch_size=batch_size, shuffle=True, train=False, transform=transform_test)

	cifar_train, cifar_sampler, cifar_test = getNormalizedCIFAR( cvs_path='data_path', batch_size=1024, shuffle=True, sampler=sampler, train_trainform=transform_train, test_transform=transform_test )

'''

print( "...done loading data")
print( len( cifar_train.dataset) )
print( len( cifar_test.dataset ) )

print
print
print

#kfac here. 
bias=False

damp_gamma  = 1e-8#Levenberg-Marquardt Dampening Term 
regLambda = 1e-8  # Regularization Term
mv_avg_theta = 0.9 #moving average -- momemtum


kfac  = KFACUtils( ksize = 5, layers = ['conv', 'conv', 'lin', 'lin', 'lin' ], batchSize = batch_size, bias=bias, debug=False, gamma = damp_gamma, theta = mv_avg_theta, regLambda = regLambda)


#model
if NETWORK == 'lenet': 
	model = LeNetCIFARKFAC( num_classes = 10, activation = ACTIVATION, batchSize=batch_size, kfac=kfac, bias=bias, r=regLambda); 
else: 
	model = AlexNetCIFAR( num_classes = 10, activation = ACTIVATION ); 
model.double ()
model.cuda ()

#criterion = nn.MSELoss(size_average=False, reduce=True)
criterion = nn.CrossEntropyLoss(size_average=False )
model.setLossFunction( criterion )
print( "...done network creation")

#Subsampled Trust Region solver here. 
trparams = SubsampledTRCGParams ()
trparams.delta = np.float64( .01 )
trparams.max_delta = np.float64( 100 )
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

model.initOffsets ()
#model.initRandomWeights ()
trsolver = SubsampledTRCGKFACFrank( model, ConjugateGradient, trparams, trstats, cifar_train, cifar_test, cifar_sampler )
print( "...created the solver")

#
# Initialize the starting point
#
#trsolver.initializeZero ()
#trsolver.initializeRandom ()
trsolver.initKaiming ()


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
