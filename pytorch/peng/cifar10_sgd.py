from __future__ import print_function
import argparse
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from nn_models import *
import resnet1 as rsn1
import resnet2 as rsn2
from kfac2 import *
from utils import *
import logging, os
from torch.distributions import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 1000)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                     help='SGD momentum (default: 0.5)')
# parser.add_argument('--delta', type=float, default=1.0, metavar='D',
#                     help='delta0 (default: 1.0)')
# parser.add_argument('--cg', type=int, default=250, metavar='CG',
#                     help='maximum cg iterations (default: 250)')
# parser.add_argument('--gamma1', type=float, default=2.0, metavar='G1',
#                     help='gamma1 (default: 2.0)')
# parser.add_argument('--rho1', type=float, default=0.8, metavar='R1',
#                     help='rho1 (default: 0.8)')
# parser.add_argument('--gamma2', type=float, default=1.2, metavar='G2',
#                     help='gamma2 (default: 1.2)')
# parser.add_argument('--rho2', type=float, default=1e-4, metavar='R2',
#                     help='rho2 (default: 1e-4)')
parser.add_argument('--model', type=str, default='LeNet',
                    help='neural network model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='cifar10.log' )
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--da', type=int, default=0)
parser.add_argument('--init', type=str, default='def',
                    help='init network model')
parser.add_argument('--decay-epoch', type=int, nargs='+', default=[99, 199, 299], 
                    help='learning rate decay epoch')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.log = '%s/sgd_%s_%s_%s_%s_%s%s_%s' % (args.model, args.model, str(args.batch_size), str(args.lr), str(args.da), args.init, str(args.seed), args.log)
if not os.path.exists(args.model):
    os.makedirs(args.model)

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(filename=args.log, filemode='w', format=FORMAT, level=logging.DEBUG)
logging.debug(args)
print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.da == 0:
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
else:
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


train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                   transform=transform_train),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transform_test),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
train_loader_org = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                   transform=transform_test),
    batch_size=args.batch_size, shuffle=False, **kwargs)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

n = len(train_loader.dataset)
batch_size = train_loader.batch_size

criterion = nn.CrossEntropyLoss()

model_list = {
    'QAlexNetS':QAlexNetS(),
    'QAlexNetSb':QAlexNetSb(),
    'VGG16': VGG('VGG16'),
    'VGG16b': VGGb('VGG16'),
    'ResNet18': rsn2.resnet(depth=20, num_classes=10),
    'ResNet18b': rsn1.resnet(depth=20, num_classes=10),
}
model = model_list[args.model]

if args.cuda:
    model.cuda()
    # model = torch.nn.DataParallel(model).cuda()
    
if args.init == "km":
    model.apply(init_params)
if args.init == "xavier":
    model.apply(normal_init)
if args.init == "0":
    model.apply(zeros_init)



optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


def full_pass():
    model.eval()
    loss = 0.0
    count = 0.0
    model.zero_grad()
    grads_dict = {}
    for data in train_loader:
        inputs, labels = data
#         if args.cuda:
        inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients
        # forward + backward
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        batch_loss.backward()
        count = count + 1
    for p in model.parameters():
        grads_dict[p] = p.grad.data/count
    model.train()
    return loss / count, grads_dict


def closure():
    model.eval()
    loss = 0.0
    count = 0.0
    for data in train_loader:
        inputs, labels = data
#         if args.cuda:
        inputs, labels = inputs.cuda(), labels.cuda()
        # forward
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        count = count + 1
    model.train()
    return loss / count



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
piter = args.log_interval

start_epoch = -1
fevals = 0
num_cgs = 0
num_props = 0

if args.resume:
    paths = 'model/sgd_%s_%s_%s_%s%s_%s.cpt' % (args.model, str(args.batch_size), str(args.da), args.init, str(args.seed), args.resume)
    print("loading checkpoint" + paths)
    checkpoint = torch.load(paths)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = args.resume
    
run_time = 0.0
flag = True
for epoch in range(start_epoch+1, args.epochs):  # loop over the dataset multiple times
    if epoch in args.decay_epoch:
        optimizer.param_groups[0]['lr'] /= 10.0
    beg = time.time()            
    for i, (inputs, labels) in enumerate(train_loader):
        model.train()
        if args.cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    run_time += time.time() - beg
    train_loss, train_acc = test(model, train_loader_org, criterion)
    test_loss, test_acc = test(model, test_loader, criterion)
    logging.debug("[Epoch {}] Time: {}, Train loss: {}, Train accuracy: {}, Test loss: {}, Test accuracy: {}".format(epoch, run_time,  train_loss, train_acc, test_loss, test_acc))
    # if args.save:
    #     paths = 'model/sgd_%s_%s_%s_%s_%s_%s%s_%s.cpt' % (args.model, str(args.delta),str(args.cg), str(args.batch_size), str(args.da), args.init, str(args.seed), epoch)
    #     torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()}, paths)
        
print('Finished Training')
