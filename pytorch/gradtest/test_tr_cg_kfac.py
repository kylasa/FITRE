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

from torchvision import datasets, transforms


from dataset_loader import get_tensor_data_loader 
from dataset_loader import get_cifar_data_loader
from alexnet_cifar import ALEXNETCIFAR

from utils import *

TYPE = torch.DoubleTensor


print( "======== TR CG Solver =========")

#Path variables here. 
train_data = '/scratch/skylasa/solvers/cnn-data/'
test_data = '/scratch/skylasa/solvers/cnn-data/'
data_path = '/scratch/skylasa/solvers/cnn-data/'

print( "...begin")

transform_train = transforms.Compose([
        transforms.ToTensor()
    ])  
transform_test = transforms.Compose([
        transforms.ToTensor()
    ])  

batch_size = 500

cifar_train, cifar_sampler = get_cifar_data_loader( csv_path = train_data, batch_size=batch_size, transform=transform_train)
cifar_train_test, cifar_sampler = get_cifar_data_loader( csv_path = train_data, batch_size=batch_size, train=True, transform=transform_test)
cifar_test, cifar_test_sampler = get_cifar_data_loader( csv_path=test_data, batch_size=batch_size, train=False, transform=transform_test)

	
print( "...done loading data")
print( len( cifar_train.dataset) )
print( len( cifar_test.dataset ) )

print
print
print

#kfac here. 
bias=False


model = ALEXNETCIFAR( num_classes = 10, bias=bias )
model.double ()
model.cuda ()

criterion = nn.CrossEntropyLoss()
model.setLossFunction( criterion )
print( "...done network creation")


#model.load_state_dict( torch.load( '/scratch/skylasa/peng/params_test.file' ))
p =  torch.load( '/scratch/skylasa/peng/params_test.file' )
model.setWeights( p[ 'features.0.weight' ], p[ 'features.3.weight'], p[ 'classifier.0.weight' ], p[ 'classifier.2.weight' ], p['classifier.4.weight' ] )
print( 'Done loading the parameter file.... ' )

for l in cifar_train: 

	X_sample_var = l[0].type( TYPE ).cuda ()
	Y_sample_var = l[1].type( torch.LongTensor ).cuda ()

	model.zero_grad ()
	grad = model.computeGradientIter2( X_sample_var, Y_sample_var  )

	print( 'grad norm', math.sqrt( group_product( grad, grad ) ) ) 
	print( 'weights norm', math.sqrt( group_product( model.parameters (), model.parameters () ) ) ) 

	break


