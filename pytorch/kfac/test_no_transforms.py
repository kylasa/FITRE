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
from alexnet_cifar import AlexNetCIFAR
from lenet_cifar import LeNetCIFAR

from conjugate_gradient import ConjugateGradient
from subsampled_tr_cg import SubsampledTRCG
from tr_cg_utils import SubsampledTRCGParams
from tr_cg_utils import SubsampledTRCGStatistics

print( "======== TR CG Solver =========")

#Path variables here. 
train_data = '/home/skylasa/datasets/cifar-10-batches-py/'
test_data = '/home/skylasa/datasets/cifar-10-batches-py/'
data_path = '/home/skylasa/datasets/'

DEVICE = int( sys.argv[1] )
NETWORK = sys.argv[2]
ACTIVATION = sys.argv[3]
INITIALIZATION = sys.argv[4]
LOGFILE = sys.argv[5]
SCALE = float ( sys.argv[6] )
TRANSFORMS = sys.argv[7]

print( "...begin")

torch.manual_seed( 999 )
torch.cuda.manual_seed_all( 999 )

torch.cuda.set_device( DEVICE )

#Read the dataset here. 
batch_size=1024
train_size = 50000

if TRANSFORMS == 'no': 
	sampler = WeightedRandomSampler( np.ones( train_size ) * (1./np.float(train_size)) , train_size, replacement=True )
	cifar_train, cifar_sampler = get_tensor_data_loader( cvs_path=train_data, batch_size=1024, shuffle=True, sampler=sampler)
	cifar_test, cifar_test_sampler = get_tensor_data_loader( cvs_path=test_data, fileFilter='test_batch*', batch_size=1024, shuffle=True )
	

if TRANSFORMS == 'yes': 
	transform_train = transforms.Compose([
		#transforms.RandomCrop(32, padding=4),
		#transforms.RandomHorizontalFlip(),
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
	cifar_train, cifar_sampler = get_cifar_data_loader( csv_path = data_path, batch_size=batch_size, shuffle=True, transform=transform_train, sampler=sampler)
	cifar_test, tt  = get_cifar_data_loader( csv_path = data_path, batch_size=batch_size, shuffle=True, train=False, transform=transform_test)


print( "...done loading data")
print( len( cifar_train.dataset) )
print( len( cifar_test.dataset ) )
#exit ()

#model
if NETWORK == 'lenet': 
	model = LeNetCIFAR( num_classes = 10, activation = ACTIVATION); 
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

model.initOffsets ()
#model.initRandomWeights ()
trsolver = SubsampledTRCG( model, ConjugateGradient, trparams, trstats, cifar_train, cifar_test, cifar_sampler, INITIALIZATION, SCALE )
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
