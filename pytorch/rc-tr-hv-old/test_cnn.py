__author__ = 'Sudhir Kylasa'

import os
import datetime
import sys
import numpy as np
import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.distributed as dist

from dataset_loader import get_tensor_data_loader
from dataset_loader import get_cifar_data_loader
from alexnet_cifar import AlexNetCIFAR
from lenet_cifar import LeNetCIFAR

from torchvision.transforms import transforms

torch.manual_seed( 999 )
torch.cuda.manual_seed_all( 999 )

print( sys.argv[1] )
print( sys.argv[2] )
print( sys.argv[3] )
print( sys.argv[4] )
print( sys.argv[5] )

solver = "Adam"
if sys.argv[1] == "Adam": 
	solver = "Adam"
elif sys.argv[1] == "Adagrad": 
	solver = "Adagrad"
elif sys.argv[1] == "Adadelta": 
	solver = "Adadelta"
elif sys.argv[1] == "SGD": 
	solver = "SGD"
else: 
	solver = "RMSprop"


out_dir = sys.argv[2]
which_network = sys.argv[3]
which_activation = sys.argv[4]
useTransforms = sys.argv[5]
learning_rate = float( sys.argv[6] ) 
CUDA_DEVICE_ID = int( sys.argv[7] )


torch.cuda.set_device( CUDA_DEVICE_ID )

outfile = open( out_dir + which_network + solver +'_' + which_activation + '_' +str(learning_rate) + '_' + useTransforms +'.txt', 'w' )
outfile.write('Solver: {} \n'.format(solver) )
outfile.write('Activation: %s\n' % ( which_activation ) )
outfile.write('Network : %s\n' % ( which_network ) )
outfile.write('Learning_rate : %e\n' % ( learning_rate ) )
outfile.write('DEVICE ID : %d\n' % ( CUDA_DEVICE_ID ) )
outfile.write('Apply Transforms: %s \n' % (useTransforms) )

#Hyper Parameters 
num_epochs = 500
batch_size = 256

#Read the dataset here. 
if useTransforms == 'no' : 
	data_path = '/home/skylasa/datasets/cifar-10-batches-py/'
	cifar_train, tt = get_tensor_data_loader( cvs_path=data_path, batch_size=batch_size, shuffle=True)
	cifar_test, tt = get_tensor_data_loader( cvs_path=data_path, fileFilter='test_batch*', batch_size=batch_size, shuffle=True)
else: 
	data_path = '/home/skylasa/datasets/'
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

	cifar_train, tt = get_cifar_data_loader( csv_path = data_path, batch_size=batch_size, shuffle=True, transform=transform_train )
	cifar_test, tt = get_cifar_data_loader( csv_path = data_path, batch_size=batch_size, shuffle=True, train=False, transform=transform_test )

outfile.write('Train Size: {}, Test Size: {}\n'.format(len( cifar_train.dataset ), len( cifar_test.dataset )) )

if which_network == "lenet": 
	model = LeNetCIFAR( num_classes = 10, activation=which_activation)
else: 
	model = AlexNetCIFAR(num_classes=10, activation=which_activation)
model.double ()
model.cuda ()

criterion = nn.CrossEntropyLoss (size_average=False)
model.setLossFunction( criterion )

#Optimizer to use
#Adam
if solver == "Adam": 
	optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate )

#Adadelta
if solver == "Adadelta": 
	optimizer = torch.optim.Adadelta( model.parameters (), lr=learning_rate )

#Adagrad
if solver == "Adagrad": 
	optimizer = torch.optim.Adagrad( model.parameters (), lr=learning_rate )

#RMSProp
if solver == "RMSprop": 
	optimizer = torch.optim.RMSprop( model.parameters (), lr=learning_rate )

if solver == "SGD": 
	outfile.write('We are using SGD Here.... \n')
	optimizer = torch.optim.SGD( model.parameters (), lr=learning_rate, momentum=0.9, weight_decay=5e-4 )
	


start = 0
end = 0
cumTime = 0

model.initKaimingUniform ()

#Zero the gradient
optimizer.zero_grad()

outfile.write( '************ START **********\n' )
total_time = 0

#Epochs here....
for epoch in range(num_epochs):

	train_hits = 0
	train_loss = 0
	train_ll = 0
	epoch_start = datetime.datetime.now ()

	for batch_idx, (inputs, targets) in enumerate( cifar_train ): 
		x_var = Variable( inputs.type( torch.DoubleTensor ).cuda () )
		#y_var = Variable( targets.type( torch.DoubleTensor ).cuda () )
		y_var = Variable( targets.cuda () )

		optimizer.zero_grad ()
		ll, loss, pred = model.evalModelChunk( x_var, y_var )
		#model._computeGradient( loss, True)
		model._computeGradBackward( loss )
		optimizer.step ()

		#train_hits += (pred == torch.max(targets, 1)[1].numpy ()).sum ()
		train_hits += (pred.eq(y_var)).cpu ().sum ().data[0]
		train_ll += ll
		train_loss += loss

		x_var.volatile = True
		y_var.volatile = True
		#break

	epoch_end = datetime.datetime.now ()
	epoch_time = (epoch_end - epoch_start).total_seconds ()
	total_time += epoch_time

	test_ll = 0
	test_loss = 0
	test_accu = 0
	test_ll, test_loss, test_accu = model.evalModel( cifar_test )

	#normalize
	train_accu = np.float(train_hits) * 100. / np.float( len( cifar_train.dataset ) )
	train_loss = train_loss.cpu().data[0] / np.float( len( cifar_train.dataset ) )	
	train_ll = train_ll.cpu ().data[0] / np.float( len( cifar_train.dataset ) )

	outfile.write('{}\t{:.4e}\t{:.2f}\t{:.4e}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format( epoch+1, train_loss, train_accu, test_loss, test_accu, epoch_time, total_time ) )

outfile.close ()
