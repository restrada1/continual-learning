import argparse
import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import helper_functions
import models
import pandas as pd
#import Resnet
from utils import progress_bar
import dense_net2
import os
#from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
################################################################################
# EXECUTION FLAGS                                 #                             
################################################################################
Flags = {}
Flags['CIFAR_100'] = False
Flags['CIFAR_10_Incrmental'] = True
Flags['NC_NET'] = False
Flags['NC_NET_CUM'] = False
Flags['CAE_Train'] = True
Flags['SPLIT_INPUT']=True
best_acc = 0  # best test accuracy

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

#model = Resnet.ResNet18()
net=models.Net()#dense_net2.DenseNet( growthRate=12)#Resnet.ResNet18()Resnet.Net() #
print(net)
criterion = nn.CrossEntropyLoss()
# if Flags['CIFAR_100'] :
#     optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=1e-4)
# else:
#     optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

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

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='../data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=4)


######################################################################################################
# SAVING ALL THE LABEL DATA FOR RELOADING PURPOSE
######################################################################################################
global_trainset_train_labels=trainset.train_labels
global_trainset_train_data=trainset.train_data
global_testset_test_labels=testset.test_labels
global_testset_test_data=testset.test_data
global_train_loader=train_loader
global_test_loader=test_loader
#####################################################################################################################
# Calculating all the Train and Test Indexes for Each Digit
#####################################################################################################################
train_Indexs=[[] for _ in range(0,100)]
test_Indexs = [[] for _ in range(0,100)]

for num in range(0,len(trainset.train_labels)):
    train_Indexs[trainset.train_labels[num]].append(num)
    
for num in range(0,len(testset.test_labels)):
    test_Indexs[testset.test_labels[num]].append(num)

######################################################################################################
# RELOD THE DATASET WHEN NEEDED
######################################################################################################
def RELOAD_DATASET():
    #print("RELOADING THE DATA")
    global train_loader,test_loader,trainset,testset

    train_loader=global_train_loader
    test_loader=global_test_loader

    trainset.train_data=global_trainset_train_data
    testset.test_data=global_testset_test_data

    trainset.train_labels=global_trainset_train_labels
    testset.test_labels=global_testset_test_labels

######################################################################################################
#MODIFYING DATSET FOR INDIVIDUAL CLASSES
######################################################################################################
def load_individual_class(postive_class,negative_classes):
    RELOAD_DATASET()
    global train_loader,test_loader,train_Indexs,test_Indexs
    index_train_postive=[]
    index_test_postive=[]
    index_train_negative=[]
    index_test_negative=[]
    for i in range(0,len(postive_class)):
        index_train_postive=index_train_postive+train_Indexs[postive_class[i]]
        index_test_postive=index_test_postive+test_Indexs[postive_class[i]]
        
    for i in range(0,len(negative_classes)):
        index_train_negative=index_train_negative+train_Indexs[negative_classes[i]][0:int(0.5*(len(train_Indexs[negative_classes[i]])))]
        index_test_negative=index_test_negative+test_Indexs[negative_classes[i]][0:int(0.5*(len(test_Indexs[negative_classes[i]])))]
    
    index_train=index_train_postive+index_train_negative
    index_test=index_test_postive+index_test_negative
    modified_train_labels = [int(trainset.train_labels[i]%10) for i in index_train] #[1 if (trainset.train_labels[x] in postive_class) else 0 for x in index_train]
    modified_test_labels  = [int(testset.test_labels[i]%10) for i in index_test]#[1 if (testset.test_labels[x] in postive_class) else 0 for x in index_test]
    print("Modified Test Labels are",modified_train_labels[1:100])
    trainset.train_labels=modified_train_labels#train set labels
    trainset.train_data=trainset.train_data[index_train]#train set data
    testset.test_labels=modified_test_labels#testset labels
    testset.test_data=testset.test_data[index_test]#testset data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=True) 

#####################################################################################################################
# Calculating all the Train and Test Indexes for Each Digit
#####################################################################################################################

def train(args, model, device, train_loader, optimizer, epoch):
    model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        #print("loss is ",loss.data[0])
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)#.item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

#####################################################################################################################
# Calculating all the Train and Test Indexes for Each Digit
#####################################################################################################################
def TRAIN_TEST_MNIST(epoch):
    for epoch in range(0,epoch):
        adjust_learning_rate(optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        acc=test(args, model, device, test_loader)
    return acc

#####################################################################################################################
# Calculating all the Train and Test Indexes for Each Digit
#####################################################################################################################
def TRAIN_TEST_MNIST_Without_adj_lr(model,epoch):
    for epoch in range(0,epoch):
        #adjust_learning_rate(optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        acc=test(args, model, device, test_loader)
    return acc

if Flags['CIFAR_100']:
    TRAIN_TEST_MNIST(50)

# Training
def train_cifar10(epoch):
    print('\nEpoch: %d' % epoch)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
def test_cifar10(epoch):
    global best_acc
    net.to(device)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc



if Flags['CIFAR_10_Incrmental']:
    for classess in range(0,10):
        load_individual_class(list(range(classess*10, (classess+1)*10)),[])
        for epoch in range(0,5):
            train_cifar10(epoch)
            test_cifar10(epoch)
        torch.save(net.state_dict(),'cifar_individual_weights/cifar_classes_'+str(classess*10)+'to'+str((classess+1)*10)+'.pt')
        # criterion = nn.CrossEntropyLoss()

        # #model = dense_net2.DenseNet(num_classes=10)
        # #print(list(range(0, (classess+1)*10)))
        # #print(model,"\n","--------------------")
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # accuracy=TRAIN_TEST_MNIST_Without_adj_lr(model,1)
        # Actual_Accuracy.append(int(accuracy))
        # print("Actual acc",Actual_Accuracy)
        # torch.save(model.state_dict(),'cifar_individual_weights/permutated_MNIST/mnist_digit_solver_'+str(mnsit_class)+'.pt')
        # if mnsit_class==0 :#or mnsit_class==1:
        #     FLATTEN_WEIGHTS_TRAIN_VAE([],model)
        # else:
        #     for i in range(0,x-1):  
        #         #print(i,Skill_Mu)
        #         recall_memory_sample=Skill_Mu[i][0]#Tasks_MU_LOGVAR[i]['mu']#[len(Tasks_MU_LOGVAR[i]['mu'])-1]
        #         generated_sample=student_model.decoder(recall_memory_sample).cpu()
        #         task_samples.append(generated_sample)
        #     FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,model) 
