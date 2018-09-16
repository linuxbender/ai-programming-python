import torch
from collections import OrderedDict
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
import torch.nn.functional as F

# get ModelClassifier
def getModelClassifier(inputSize, firstHiddenUnit, secHiddenUnit):
    hiddenSize = [firstHiddenUnit, secHiddenUnit]
    outputSize = 102
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(inputSize, hiddenSize[0])),
        ('drop1', nn.Dropout(p=0.5)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hiddenSize[0], hiddenSize[1])),
        ('drop2', nn.Dropout(p=0.4)),
        ('relu2', nn.ReLU()),
        ('fc4', nn.Linear(hiddenSize[1], outputSize)),    
        ('output', nn.LogSoftmax(dim=1))]))
    return classifier

# check gpu support
def gpuSupport(gpu):
    if gpu and torch.cuda.is_available():        
        print('\nRunning GPU...\n')
        return True
    return False

# get model size from arch
def getModelSizeFromArch(arch):   
    if arch == 'vgg13':
        return 25088
    if arch == 'densenet121':
        return 1024
    return 0