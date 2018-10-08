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
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import helper_functions
import models
import time
import pandas as pd
import utils
import numpy as np
import os
from PIL import Image
import pickle
#from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
################################################################################
#                              EXECUTION FLAGS                                 #                             
################################################################################
Flags = {}
Flags['CIFAR_100'] = False
Flags['CIFAR_10_Incrmental'] = True
Flags['NC_NET'] = False
Flags['NC_NET_CUM'] = False
Flags['CAE_Train'] = True
Flags['SPLIT_INPUT']=True
best_acc = 0  # best test accuracy

######################################################################################################
#                                   PARSER ARGUMENTS
######################################################################################################
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
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


######################################################################################################
#                                   LOAD DATASET AND TRANSFORMS
######################################################################################################
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

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

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='../data', train=True,
                                        download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                       download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=4)


######################################################################################################
#                        SAVING ALL THE LABEL DATA FOR RELOADING PURPOSE
######################################################################################################
global_trainset_train_labels=trainset.train_labels
global_trainset_train_data=trainset.train_data
global_testset_test_labels=testset.test_labels
global_testset_test_data=testset.test_data
global_train_loader=train_loader
global_test_loader=test_loader
#####################################################################################################################
#                       Calculating all the Train and Test Indexes for Each Digit
#####################################################################################################################
train_Indexs=[[] for _ in range(0,100)]
test_Indexs = [[] for _ in range(0,100)]

for num in range(0,len(trainset.train_labels)):
    train_Indexs[trainset.train_labels[num]].append(num)
    
for num in range(0,len(testset.test_labels)):
    test_Indexs[testset.test_labels[num]].append(num)


######################################################################################################
#                                   RELOD THE DATASET WHEN NEEDED
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
#                                   RELOD THE WEIGHTS INTO NET                                       #
######################################################################################################
def loadWeights_cifar(weights_to_load, net):
    net.conv1.weight.data = torch.from_numpy(weights_to_load[0])
    net.conv1.bias.data =   torch.from_numpy(weights_to_load[1])
    net.conv2.weight.data = torch.from_numpy(weights_to_load[2])
    net.conv2.bias.data =   torch.from_numpy(weights_to_load[3])
    net.fc1.weight.data =   torch.from_numpy(weights_to_load[4])
    net.fc1.bias.data =     torch.from_numpy(weights_to_load[5])
    net.fc2.weight.data =   torch.from_numpy(weights_to_load[6])
    net.fc2.bias.data =     torch.from_numpy(weights_to_load[7])
    net.fc3.weight.data =   torch.from_numpy(weights_to_load[8])
    net.fc3.bias.data =     torch.from_numpy(weights_to_load[9])
    return net

######################################################################################################
#                                       PLOTTING                                                     #
######################################################################################################
def plotAccuracies(allAcc):
    for s,acc in enumerate(allAcc):
        x = np.array(acc)
        f = plt.figure()
        for i in range(x.shape[1]):
            plt.plot(x[:,i],'.-', markersize=5, label='Skill: '+str(i))
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Encoded Skills -- Stage ' + str(s))
        plt.legend(loc='lower right')
        plt.show()    
        f.savefig('./CIFAR_plots/acc_'+str(s)+'.pdf', bbox_inches='tight')
        plt.close()


def plotAccuracyDiff(allAcc,Actual_Accuracies):
    for s,acc in enumerate(allAcc):
        x = np.array(acc)
        f = plt.figure()
        for i in range(x.shape[1]):
            plt.plot(Actual_Accuracies[i] - x[:,i],'.-', markersize=5, 
                label='Skill: '+str(i))
        plt.xlabel('Iterations')
        plt.ylabel('True Acc - Reconst. Acc')
        plt.title('Encoded Skills -- Stage ' + str(s))
        plt.legend(loc='upper right')
        plt.show()    
        f.savefig('./CIFAR_plots/acc_diff_'+str(s)+'.pdf', bbox_inches='tight')
        plt.close()


def plotAccuracyDiffPoint(allAcc,Actual_Accuracies,testPoint=25):
    for s,acc in enumerate(allAcc):
        x = np.array(acc)
        f = plt.figure()
        for i in range(x.shape[1]):
            plt.plot(i,Actual_Accuracies[i] - x[testPoint,i],'.-', markersize=10, 
                label='Skill: '+str(i))
        plt.xlabel('Iterations')
        plt.ylabel('True Acc - Reconst. Acc')
        plt.title('Encoded Skills -- Stage ' + str(s) + ', Iter ' + str(testPoint))
        plt.legend(loc='lower right')
        plt.show()    
        f.savefig('./CIFAR_plots/acc_diff_pt_'+str(s)+'_'+str(testPoint)+'.pdf', bbox_inches='tight')
        plt.close()

def plotAccuracyDiffPointAll(allAcc,Actual_Accuracies,testPoint=25):
    f = plt.figure()
    yAll = []
    for i in range(10):
        yAll.append([])
        for s in range(i,10):
            yAll[i].append(Actual_Accuracies[i] - allAcc[s][testPoint][i])
    
    for i,y in enumerate(yAll):
        plt.plot(range(i,10),y,'.-', markersize=5,label='Skill: '+str(i))
    plt.xlabel('Stages')
    plt.ylabel('True Acc - Reconst. Acc')
    plt.title('Encoded Skills -- Iter ' + str(testPoint))
    plt.legend(loc='upper left')
    plt.show()    
    f.savefig('./CIFAR_plots/acc_diff_pt_all_'+str(s)+'_'+str(testPoint)+'.pdf', bbox_inches='tight')
    plt.close()

######################################################################################################
#                           MODIFYING DATSET FOR 10 CLASSES
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
    #print("Modified Test Labels are",modified_train_labels[1:100])
    trainset.train_labels=modified_train_labels#train set labels
    trainset.train_data=trainset.train_data[index_train]#train set data
    testset.test_labels=modified_test_labels#testset labels
    testset.test_data=testset.test_data[index_test]#testset data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=True) 

# CALUCULATE_TRAIN_INDEXES()
# CALUCULATE_TEST_INDEXES()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('train_data_image.png')

def imshow_test(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('test_data_image.png')

def SHOW_TEST_TRAIN_IMAGES_SAMPLE():
    # get some random training images
    global train_loader
    global test_loader
    dataiter = iter(train_loader)
    dataiter_test = iter(test_loader)
    #print("length of testset loader is",len(test_loader))
    #print("length if train set loader is",len(train_loader))
    images, labels = dataiter.next()
    images_test, labels_test = dataiter_test.next()
    imshow(torchvision.utils.make_grid(images))
    imshow_test(torchvision.utils.make_grid(images_test))
    img = Image.open('train_data_image.png')
    img.show()
    time.sleep(5)
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    img = Image.open('test_data_image.png')
    print(' '.join('%5s' % classes[labels_test[j]] for j in range(4)))
    img.show() 


#####################################################################################################################
#                                              TRAIN CIFAR WITHOUT PROGRESS BAR                                     #
#####################################################################################################################

def train(epochs):
    net.to(device)
    net.train()
    for epoch in range(0,epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            #print("loss is ",loss.data[0])
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        accuracy=test()
    return accuracy

#####################################################################################################################
#                                               TRAIN CIFAR WITHOUT PROGRESS BAR                                    #
#####################################################################################################################

def test():
    net.to(device)
    net.eval()
    test_loss = 0
    correct = 0
    #print("length of testset loader is",len(test_loader))
    #print("length if train set loader is",len(train_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target)#.item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

#####################################################################################################################
#                                  TRAIN CIFAR WITH PROGRESS BAR                                                    #
#####################################################################################################################
# Training
def train_cifar10(epoch):
    print('\nEpoch: %d' % epoch)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #criterion = nn.CrossEntropyLoss()
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

        utils.progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

#####################################################################################################################
#                                  TEST CIFAR WITH PROGRESS BAR                                                    #
#####################################################################################################################
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

            utils.progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
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
    return acc


######################################################################################################
#                           FLATTEN THE CIFAR WEIGHTS AND FEED IT TO CAE                             #
######################################################################################################
def FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,model):
    final_skill_sample=[]
    Flat_input,net_shapes=helper_functions.flattenNetwork(model.cpu())
    final_skill_sample.append(Flat_input)
    if len(task_samples)==0:
        accuracies = CAE_AE_TRAIN(net_shapes,task_samples+final_skill_sample,100)
    else:
        accuracies = CAE_AE_TRAIN(net_shapes,task_samples+final_skill_sample,300)
    return accuracies

######################################################################################################
#                                   GLOBAL VARIABLES
######################################################################################################
#net=models.Net().to(device)
net_reset=models.Net().to(device)
Actual_Accuracy=[]
criterion = nn.CrossEntropyLoss()
student_model=models.CAE()#.to(device)
teacher_model=models.CAE().to(device)
vae_optimizer = optim.Adam(student_model.parameters(), lr = 0.0001)#, amsgrad=True)
lam = 0.001
Actual_Accuracy=[]
threshold_batchid=[]
#biased training variables
nSamples = 10
nBiased = min(nSamples,10)
trainBias = 0.5
minReps = 1
nReps = 30
stage=0
addRepsTotal = nReps*minReps
#####################################################################################################################
#                                              CAE TRAIN AND CIFAR TEST                                             #
#####################################################################################################################
print("VAE SCHEDULER IS ", vae_optimizer)
Skill_Mu=[]
def CAE_AE_TRAIN(shapes,task_samples,iterations):
    global stage
    student_model.train()
    print("len of takssample is",len(task_samples))
    accuracies = np.zeros((iterations,len(task_samples)))
    for i in range(0,len(task_samples)-len(Skill_Mu)):
        Skill_Mu.append([])
    for batch_idx in range(1,iterations):
        #randPerm = np.random.permutation(len(task_samples))
        randPerm,nReps = helper_functions.biased_permutation(stage+1,nBiased,trainBias,len(task_samples),minReps)
        print(randPerm,nReps)
        for s in randPerm:
            skill=s
            vae_optimizer.zero_grad()
            data=Variable(torch.FloatTensor(task_samples[skill]))#.to(device)
            hidden_representation, recons_x = student_model(data)
            W = student_model.state_dict()['fc12.weight']
            loss,con_mse,con_loss,closs_mul_lam = helper_functions.Contractive_loss_function(W, 
                data.view(-1, 62006), recons_x,hidden_representation, lam)
            Skill_Mu[skill]=hidden_representation
            loss.backward()
            vae_optimizer.step()
            print('Train Iteration: {},tLoss: {:.6f},picked skill {}'.format(batch_idx,loss.data[0],skill ))
        
        if batch_idx %1==0:
            values=0
            for i in range(0,len(task_samples)):
                mu1=Skill_Mu[i][0]
                task_sample = student_model.decoder(mu1).cpu()
                sample=task_sample.data.numpy().reshape(62006)
                #print('MSE IS',i,mean_squared_error(task_sample.data.numpy(),Variable(torch.FloatTensor(task_samples[i])).data.numpy()))
                #mse=mean_squared_error(task_sample.data.numpy(),Variable(torch.FloatTensor(task_samples[i])).data.numpy())
                final_weights=helper_functions.unFlattenNetwork(sample, shapes)
                loadWeights_cifar(final_weights,net)
                load_individual_class(list(range(i*10, (i+1)*10)),[]) 
                Avg_Accuracy=test()
                accuracies[batch_idx,i] = Avg_Accuracy
                if round(Avg_Accuracy+0.5)>=int(Actual_Accuracy[i]):
                    values=values+1
            if values==len(task_samples):
                print("########## \n Batch id is",batch_idx,"\n#########")
                threshold_batchid.append(batch_idx)
                break
    stage=stage+1
    return accuracies
#####################################################################################################################
#                                  CIFAR AND CAE TRAIN START HERE ALL START HERE                                    #
#####################################################################################################################
#if Flags['CIFAR_10_Incrmental']:
for cifar_class in range(0,10):
    allAcc = []
    net=models.Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    task_samples=[]
    load_individual_class(list(range(cifar_class*10, (cifar_class+1)*10)),[])
    #SHOW_TEST_TRAIN_IMAGES_SAMPLE()
    accuracy=train(6)
    torch.save(net.state_dict(),'cifar_individual_weights/cifar_classes_'+str(cifar_class)+'.pt')
    Actual_Accuracy.append(int(accuracy))
    print("Actual ACCCC",Actual_Accuracy,cifar_class)
    if cifar_class==0:
        accuracies=FLATTEN_WEIGHTS_TRAIN_VAE([],net)
    else:
        print("CAMMME")
        for i in range(0, cifar_class):
            print("Recollecting the Memory Samples")
            recall_memory_sample=Skill_Mu[i][0]
            generated_sample=student_model.decoder(recall_memory_sample).cpu()
            task_samples.append(generated_sample)
        accuracies = FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,net) 
    allAcc.append(accuracies)

with open('allAcc', 'wb') as fp:
    pickle.dump(allAcc, fp)

with open('Actual_Accuracy', 'wb') as fp:
    pickle.dump(Actual_Accuracy, fp)

plotAccuracies(allAcc)
plotAccuracyDiff(allAcc,Actual_Accuracy)
plotAccuracyDiffPoint(allAcc,Actual_Accuracy,testPoint=25)
plotAccuracyDiffPointAll(allAcc,Actual_Accuracy,testPoint=25)