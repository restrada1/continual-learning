"""
SETUP:
    IDENTICAL TO Exp1_inc_mnist.py except using a modified loss for CAE

AIM:  
    CHECK IF USING THE FISHER_WEIGHTED_MSE ALLOWS THE CAE TO TRAIN FASTER AND 
    WHETHER IT IS A MORE ACCURATE GAUGE OF RECONSTRUCTED NETWORK PERFORMANCE


FINDINGS: 


"""


from __future__ import print_function
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import models
import helper_functions
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from scipy.stats import geom
# Execution flags
Flags = {}
Flags['mnist_train'] =  True #This will run the mnist experiment then run the incremental approach
#else it will load the existing weights in the directory and run the experiment
Flags['regular_mnist_train']= True # For choosing the  incase

nSamples = 10
nBiased = min(nSamples,10)
trainBias = 0.5
minReps = 1
nReps = 20
stage=0
fisher_matrix=[]
addRepsTotal = nReps*minReps
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--vaelr', type=float, default=0.001, metavar='VAE_LR',
                    help='vae learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--schedule', type=int, nargs='+', default=[20,80],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cpu") #"cuda" if use_cuda else 
#idx_permute = torch.from_numpy(np.random.permutation(784), dtype=torch.int64)
idx_permute = [np.random.permutation(28**2) for _ in range(1)] 
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),
              transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28) )])

kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
trainset=datasets.MNIST(root='./data', train=True,download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False,download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=8,shuffle=True,**kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=True, **kwargs)
global_trainset_train_labels=trainset.train_labels
global_trainset_train_data=trainset.train_data
global_testset_test_labels=testset.test_labels
global_testset_test_data=testset.test_data
global_train_loader=train_loader
global_test_loader=test_loader


train_Indexs=[[]] * 10
test_Indexs = [[]] * 10

#####################################################################################################################
# Calculating all the Train Indexes for Each Digit
#####################################################################################################################
def CALUCULATE_TRAIN_INDEXES():
    train_Indexs[0]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 0]
    train_Indexs[1]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 1]
    train_Indexs[2]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 2]
    train_Indexs[3]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 3]
    train_Indexs[4]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 4]
    train_Indexs[5]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 5]
    train_Indexs[6]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 6]
    train_Indexs[7]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 7]
    train_Indexs[8]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 8]
    train_Indexs[9]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 9]
#####################################################################################################################
# Calculating all the Test Indexes for Each Digit
#####################################################################################################################
def CALUCULATE_TEST_INDEXES():
    test_Indexs[0]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 0]
    test_Indexs[1]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 1]
    test_Indexs[2]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 2]
    test_Indexs[3]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 3]
    test_Indexs[4]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 4]
    test_Indexs[5]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 5]
    test_Indexs[6]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 6]
    test_Indexs[7]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 7]
    test_Indexs[8]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 8]
    test_Indexs[9]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 9]
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
    #print(postive_class)
    for i in range(0,len(postive_class)):
        index_train_postive=index_train_postive+train_Indexs[postive_class[i]]
        index_test_postive=index_test_postive+test_Indexs[postive_class[i]]
        
    for i in range(0,len(negative_classes)):
        index_train_negative=index_train_negative+train_Indexs[negative_classes[i]][0:int(0.2*(len(train_Indexs[negative_classes[i]])))]
        index_test_negative=index_test_negative+test_Indexs[negative_classes[i]][0:int(0.2*(len(test_Indexs[negative_classes[i]])))]
    
    index_train=index_train_postive+index_train_negative
    index_test=index_test_postive+index_test_negative
    modified_train_labels = [1 if (trainset.train_labels[x] in postive_class) else 0 for x in index_train]
    modified_test_labels  = [1 if (testset.test_labels[x] in postive_class) else 0 for x in index_test]
    trainset.train_labels=modified_train_labels#train set labels
    trainset.train_data=trainset.train_data[index_train]#train set data
    testset.test_labels=modified_test_labels#testset labels
    testset.test_data=testset.test_data[index_test]#testset data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=8,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=True) 

CALUCULATE_TRAIN_INDEXES()
CALUCULATE_TEST_INDEXES()

######################################################################################################
# LOADING OF VAE OUTPUT SKILL WEIGHTS BACK INTO THE MNSIT NETWORK
######################################################################################################
def loadWeights_mnsit(weights_to_load, net):
    model.conv1.weight.data = torch.from_numpy(weights_to_load[0]).to(device)
    model.conv1.bias.data =   torch.from_numpy(weights_to_load[1]).to(device)
    model.conv2.weight.data = torch.from_numpy(weights_to_load[2]).to(device)
    model.conv2.bias.data =   torch.from_numpy(weights_to_load[3]).to(device)
    model.fc1.weight.data =   torch.from_numpy(weights_to_load[4]).to(device)
    model.fc1.bias.data =     torch.from_numpy(weights_to_load[5]).to(device)
    model.fc2.weight.data =   torch.from_numpy(weights_to_load[6]).to(device)
    model.fc2.bias.data =     torch.from_numpy(weights_to_load[7]).to(device)
    return model

print("device is ",device)
model = models.Net().to(device)
model_reset=model
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
student_model=models.CAE().to(device)#nn.DataParallel(models.CAE().to(device))
teacher_model=models.CAE().to(device)#nn.DataParallel(models.CAE().to(device))
vae_optimizer = optim.Adam(student_model.parameters(), lr = 0.001)
lam = 0.001
Actual_Accuracy=[[] for x in range(10)]

######################################################################################################
# Adjusting Learning rate
######################################################################################################
def adjust_learning_rate(optimizer, epoch):
    global state
    #global vae_optimizer
    if epoch in args.schedule:
        state['vaelr'] *= args.gamma
        print("state lr is",state['vaelr'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['vaelr']
        print("VAE OPTIMIZER IS",vae_optimizer)
######################################################################################################
# TRAIN
######################################################################################################
def train(args, model, device, train_loader, optimizer, epoch, threshold=None):
    model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        # acc=test(args, model, device, test_loader)
        # if acc >= thresh and acc < thresh+5:
        #     break
        if int(100. * batch_idx / len(train_loader)) == 30:
            break 

        

######################################################################################################
# TEST
######################################################################################################
def test(args, model, device, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
    

######################################################################################################
# TRAIN AND TEST
######################################################################################################
def TRAIN_TEST_MNIST(thresh):
    for epoch in range(1, 2):
        train(args, model, device, train_loader, optimizer, epoch, threshold=thresh)
        acc=test(args, model, device, test_loader)
        
    return acc
######################################################################################################
# FLATTEN THE MNIST WEIGHTS AND FEED IT TO VAE
######################################################################################################
def FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,model):
    final_skill_sample=[]
    Flat_input,net_shapes=helper_functions.flattenNetwork(model.cpu())
    final_skill_sample.append(Flat_input)
    if len(task_samples) < 7:
        CAE_AE_TRAIN(net_shapes,task_samples+final_skill_sample,200)
    else:
        CAE_AE_TRAIN(net_shapes,task_samples+final_skill_sample,200)

######################################################################################
# COMPUTE THE FISHER AND USE THAT TO TRAIN THE CAE 
######################################################################################
def compute_fisher(trainData, net, fisher_samples):
    losses = [0.0 for x in range(fisher_samples)]
    lossCount = 0
    for batch_idx, (data, target) in enumerate(trainData):
        if batch_idx%5 == 0:
            data, target = data.to(device), target.to(device)
            output = net(data)
            losses[lossCount] = F.nll_loss(output, target)
            lossCount = lossCount +1
        if lossCount > fisher_samples-1:
            break
    grads = []
    for loss in losses:
        curr_grads = torch.autograd.grad(loss, net.parameters()) 
        squared_grads = []
    # print(curr_grads[0])
    for idx, grad in enumerate(curr_grads):
        squared_grads.append(grad.cpu().data.numpy()**2) #square the gradients
        grads.append(squared_grads)
    # print(grads[49][0])
    grads = np.array(grads)
    # print(grads.shape)
    
    #grads is now a 50X8 matrix with 50 samples of (squared)gradients for 8 layers
    #Now, for every parameter in the net, take the mean over the 50 samples.
    # print("sample 1: ", grads[0][1])
    # print("sample 2: ", grads[1][1])
    F_A = np.mean(grads, 0) #take the mean of the squared gradients, have confirmed this is correct
    # print("avergae: ", F_A[1])
    Fisher_Flatten=[]
    for _ in range(len(F_A)):
        #print("Shape of the fisher is",F_A[_].flatten())
        Fisher_Flatten=np.concatenate((Fisher_Flatten,F_A[_].flatten()),axis=None)
    #print("RETURNING THE FISHER")
    return Fisher_Flatten



######################################################################################################
# LOADING OF VAE OUTPUT SKILL WEIGHTS BACK INTO THE MNSIT NETWORK AND TEST MNIST
######################################################################################################
print("VAE SCHEDULER IS ", vae_optimizer)
Skill_Mu=[]
initial_fmse_threshold = 0.0002
running_thresholds = [initial_fmse_threshold for x in range(10)]
running_losses = [10.0 for x in range(10)]
running_update_counts = [0.0 for x in range(10)]
total_update_count = 0.0
average_relative_accuracies = []
def CAE_AE_TRAIN(shapes,task_samples,iterations):
    global stage
    encoded_task=[]
    global running_losses
    global running_thresholds
    global initial_fmse_threshold
    global total_update_count
    global running_update_counts
    global average_relative_accuracies
    #task_samples=task_samples.to(device)
    student_model.train()
    final_dataframe=pd.DataFrame()
    final_dataframe_1=pd.DataFrame()
    for i in range(0,len(task_samples)-len(Skill_Mu)):
        Skill_Mu.append([])
    curr_batch_avg_loss = 10
    batch_idx = -1
    curr_batch_max_loss = torch.Tensor([10.0])
    
    #while the thresholds have not been met, train
    # while np.all(running_losses[:-(10-len(task_samples))] >= running_thresholds[:-(10-len(task_samples))]):
    # while curr_batch_avg_loss >= initial_fmse_threshold:
    for batch_idx in range(1,iterations):
        
        batch_idx += 1
        train_loss = 0
        randPerm = np.random.permutation(len(task_samples))
        # adjust_learning_rate(vae_optimizer, batch_idx)
        #print("after lr change",vae_optimizer)
        # randPerm,nReps = helper_functions.biased_permutation(stage+1,nBiased,trainBias,len(task_samples),minReps)
        print(randPerm)
        #sys.exit() 
        # avg_loss = torch.Tensor([0.0])
        total_loss = torch.Tensor([0.0])
        batch_losses = [0.0 for x in range(len(task_samples))]
        for s in randPerm:
            
            total_update_count += 1
            
            collect_data=[]
            skill=s#randint(0,len(task_samples)-1)
            F_s = fisher_matrix[s]
            vae_optimizer.zero_grad()
            data=Variable(torch.FloatTensor(task_samples[skill])).to(device)
            hidden_representation, recons_x = student_model(data)
            W = student_model.state_dict()['fc2.weight']
            loss,con_mse,con_loss,closs_mul_lam = helper_functions.Contractive_loss_function(W, data.view(-1, 21432), recons_x,hidden_representation, lam)
            # loss,con_mse,con_loss,closs_mul_lam = helper_functions.Contractive_FMSE(W, data.view(-1, 21432), recons_x,hidden_representation, lam, F_s)
            # loss = loss*((total_update_count-running_update_counts[s])/total_update_count)
            
            running_update_counts[s] += 1
            Skill_Mu[skill]=hidden_representation
            batch_losses[s] = loss #update the current loss for this network
            # avg_loss += loss
            total_loss += loss
            print('Train Iteration: {},tLoss: {:.6f},picked skill {}'.format(batch_idx,loss.data[0],skill ))
        # avg_loss /= len(task_samples)
        curr_batch_avg_loss = total_loss/len(task_samples)

        #get the temporary running losses if we stopped training right here
        tmp_running_losses = [0.0 for x in range(len(task_samples))]
        for idx, x in enumerate(batch_losses):
            tmp_running_losses[idx] = (running_losses[idx] + x)-10

        


        #we need to train more on the skill where the temporary running loss deviates the most from the running threshold
        losses_wrt_thresholds = [0.0 for x in range(len(task_samples))]
        for idx,x in enumerate(batch_losses):
            
            losses_wrt_thresholds[idx] = round((batch_losses[idx].data.numpy()/running_thresholds[idx])[0], 2)

        curr_batch_max_loss = max(batch_losses)
        
        curr_batch_avg_loss.backward()
        vae_optimizer.step()

        if batch_idx %199==0:  
            # accuray_threshlod=[]
            values=0
            initial_fmse_threshold += 0.0004
            avg_relative_accuracy = 0.0
            for i in range(0,len(task_samples)):
                # Avg_Accuracy=0
                #model=models.Net()
                collect_data_1=[]
                mu1=Skill_Mu[i][0]
                task_sample = student_model.decoder(mu1).cpu()
                sample=task_sample.data.numpy().reshape(21432)
                #print('MSE IS',mean_squared_error(task_sample.data.numpy(),Variable(torch.FloatTensor(task_samples[i])).data.numpy()))
                #mse=mean_squared_error(task_sample.data.numpy(),Variable(torch.FloatTensor(task_samples[i])).data.numpy())
                final_weights=helper_functions.unFlattenNetwork(sample, shapes)
                loadWeights_mnsit(final_weights,model)
                if i%2==0:
                    load_individual_class([i],[1,3,5,7])
                else:
                    load_individual_class([i],[0,2,4,6])
                Avg_Accuracy= test(args, model, device, test_loader)
                if len(task_samples) == 10:
                    avg_relative_accuracy += round((Avg_Accuracy/Actual_Accuracy[i][len(Actual_Accuracy[i])-1])*100, 2)
                print("ORIGINAL PERFORMANCE: "+str(Actual_Accuracy[i][len(Actual_Accuracy[i])-1])+"%\n")
                print("RELATIVE PERFORMANCE TO ORIGINAL: "+str(round((Avg_Accuracy/Actual_Accuracy[i][len(Actual_Accuracy[i])-1])*100, 2))+"%\n")
                print("LOSS: ", batch_losses[i])
                # if round(Avg_Accuracy+0.5)>=int(Actual_Accuracy[i]):
                values=values+1
                #update running_thresholds and running losses
                running_losses[i] += batch_losses[i]
                
                # if len(task_samples)>=3:
                #     inc =  -(i-(len(task_samples)+1))
                #     running_thresholds[i] = initial_fmse_threshold*np.log(max(3,inc))
                # if len(task_samples) > 2:
                #     running_thresholds = [initial_fmse_threshold*np.log(len(task_samples)*1.3) for x in range(10)]
            print(running_thresholds)
            if len(task_samples) == 10:
                avg_relative_accuracy /= len(task_samples)
                average_relative_accuracies.append(avg_relative_accuracy)
            if values==len(task_samples) and len(task_samples) == 10:
                print("########## \n Batch id is",batch_idx,"\n#########")
                break
       
        
        
        
    stage=stage+1
 
avg_actual_accuracies = []
if Flags['mnist_train']:
    taskNet_thresholds = [95, 90, 85, 80, 75, 70, 65, 60]
    #for each threshold
    for thresh in taskNet_thresholds:
        avg_actual_accuracy = 0.0
        for mnsit_class in range(0,10):
            task_samples=[]
            model=model_reset
            if mnsit_class%2==0:
                load_individual_class([mnsit_class],[1,3,5,7])
            else:
                load_individual_class([mnsit_class],[0,2,4,6])
            accuracy=TRAIN_TEST_MNIST(thresh)
            avg_actual_accuracy += accuracy
            Actual_Accuracy[mnsit_class].append(int(accuracy))
            fisher_matrix.append(compute_fisher(train_loader, model, 100))
            print("Actual acc",Actual_Accuracy[mnsit_class])
            torch.save(model.state_dict(),'mnist_individual_weights/regular_MNIST/mnist_digit_solver_'+str(mnsit_class)+'.pt')
            if mnsit_class==0 :#or mnsit_class==1:
                FLATTEN_WEIGHTS_TRAIN_VAE([],model)
            else:
                for i in range(0,mnsit_class):  
                    recall_memory_sample=Skill_Mu[i][0]#Tasks_MU_LOGVAR[i]['mu']#[len(Tasks_MU_LOGVAR[i]['mu'])-1]
                    generated_sample=student_model.decoder(recall_memory_sample).cpu()
                    task_samples.append(generated_sample)
                FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,model) 
        avg_actual_accuracy /= 10
        avg_actual_accuracies.append(avg_actual_accuracy)
    print("Average Actual Accuracies: ", avg_actual_accuracies)
    print("Average Relative Accuracies: ", average_relative_accuracies)
        
else:
    if Flags['regular_mnist_train']:
        dname = 'mnist_individual_weights/regular_MNIST/'
    else:
        dname = 'mnist_individual_weights/permutated_MNIST/'
    fnames = sorted(os.listdir(dname))#getting all the saved weights in directory
    for mnsit_class in range(0,10):
        task_samples=[]
        l=[]
        l.append(mnsit_class)
        if mnsit_class%2==0:
            load_individual_class(l,[1,3,5,7])
        else:
            load_individual_class(l,[0,2,4,6])

        for i,name in enumerate(fnames):
            weights = torch.load(dname + name)
        print(dname+fnames[mnsit_class])
        model.load_state_dict(dname+fnames[mnsit_class])  
        accuracy=TRAIN_TEST_MNIST()
        Actual_Accuracy[mnsit_class].append(int(accuracy))
        print("Actual acc",Actual_Accuracy[mnsit_class])
        if mnsit_class==0 :#or mnsit_class==1:
            FLATTEN_WEIGHTS_TRAIN_VAE([],model)
        else:
            for i in range(0,len(task_samples)):  
                recall_memory_sample=Skill_Mu[i][0]#Tasks_MU_LOGVAR[i]['mu']#[len(Tasks_MU_LOGVAR[i]['mu'])-1]
                generated_sample=student_model.decoder(recall_memory_sample).cpu()
                task_samples.append(generated_sample)
            FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,model) 

