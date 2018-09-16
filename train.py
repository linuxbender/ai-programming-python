# imports 
import argparse
import numpy as np
import torch
from collections import OrderedDict

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from appHelper import gpuSupport, getModelClassifier, getModelSizeFromArch

# app logo
appLogo = """\
                     _             _       
                    | |           (_)      
  __ _ _ __  _ __   | |_ _ __ __ _ _ _ __  
 / _` | '_ \| '_ \  | __| '__/ _` | | '_ \ 
| (_| | |_) | |_) | | |_| | | (_| | | | | |
 \__,_| .__/| .__/   \__|_|  \__,_|_|_| |_|
      | |   | |                            
      |_|   |_|         
"""
# read input args from command line
def readInputArgs():
    parser = argparse.ArgumentParser(description="Training process")    
    parser.add_argument('--arch', dest='arch', default='vgg13', choices=['vgg13', 'densenet121'])
    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--firstHiddenUnit', dest='firstHiddenUnit', type=int, default=128)
    parser.add_argument('--secHiddenUnit', dest='secHiddenUnit', type=int, default=96)
    parser.add_argument('--batchSize', dest='batchSize', type=int, default=64)
    parser.add_argument('--epochs', dest='epochs', type=int, default=4)
    parser.add_argument('--checkPointFileName', dest='checkPointFileName', default='appCheckpoint.pth')    
    parser.add_argument('--gpu', dest='gpu', type=bool, default=True)
    return parser.parse_args()

# train
def train(model, epochs, criterion, optimizer, dataloaders, gpuActiv):
    if gpuActiv:
        model.cuda()
        model.to('cuda')
    else:
        model.cpu()
        
    print_every = 5
    steps = 0
    loss_show=[]    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputTrain, labelTrain) in enumerate(dataloaders['train']):            
            steps += 1
            if gpuActiv:
                inputTrain,labelTrain = inputTrain.to('cuda'), labelTrain.to('cuda')
            
            optimizer.zero_grad()            
            outputTrain = model.forward(inputTrain)
            loss = criterion(outputTrain, labelTrain)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy = 0
                for ii, (inputValid, labelValid) in enumerate(dataloaders['valid']):
                    optimizer.zero_grad()
                    if gpuActiv:
                        inputValid, labelValid = inputValid.to('cuda') , labelValid.to('cuda')
                        model.to('cuda')
                        
                    with torch.no_grad():    
                        outputValid = model.forward(inputValid)
                        vlost = criterion(outputValid,labelValid)
                        ps = torch.exp(outputValid).data
                        equality = (labelValid.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                vlost = vlost / len(dataloaders['valid'])
                accuracy = accuracy /len(dataloaders['valid'])

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost),
                       "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0  

# save model - checkPoint
def saveCheckpoint(args, model, optimizer, classifier):    
    checkpoint = {'arch': args.arch,
                  'lr': args.lr,
                  'firstHiddenUnit': args.firstHiddenUnit,
                  'secHiddenUnit': args.secHiddenUnit,
                  'batchSize': args.batchSize,
                  'epochs': args.epochs,
                  'checkPointFileName': args.checkPointFileName,
                  'gpu': args.gpu,
                  'classifier' : classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'model': model
                 }
    torch.save(checkpoint, args.checkPointFileName)
    
# main process
def main():
    print(appLogo)
    # read args
    args = readInputArgs()
    
    # file setup
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # transforms for the training, validation, and testing sets
    BATCH_SIZE = args.batchSize
    SHUFFLE = True
    NUMBER_224 = 224
    NUMBER_256 = 256
    
    data_transforms = {
    'train' : transforms.Compose([transforms.RandomResizedCrop(NUMBER_224),transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),                                                            
    'valid' : transforms.Compose([transforms.Resize(NUMBER_256),transforms.CenterCrop(NUMBER_224),transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
    'test' : transforms.Compose([transforms.Resize(NUMBER_256),transforms.CenterCrop(NUMBER_224),transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    }
    # Load the datasets with ImageFolder
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=SHUFFLE),
        'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=SHUFFLE),
        'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE, shuffle=SHUFFLE)   
    }
    
    model = getattr(models, args.arch)(pretrained=True)    
    for param in model.parameters():
        param.requires_grad = False

    #select arch modelSizeFromArch
    modelSize = getModelSizeFromArch(args.arch)
        
    # get ModelClassifier
    classifier = getModelClassifier(modelSize, args.firstHiddenUnit, args.secHiddenUnit)
    model.classifier = classifier
    
    # criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    # check gpu support
    gpuActiv = gpuSupport(args.gpu)
    
    # train the model
    print('Start training...\n')
    train(model, args.epochs, criterion, optimizer, dataloaders, gpuActiv)   
    model.class_to_idx = image_datasets['train'].class_to_idx
    # save
    print('Saving data...\n')
    saveCheckpoint(args, model, optimizer, classifier)
    print('thx and bye bye')

if __name__ == "__main__":    
    main()