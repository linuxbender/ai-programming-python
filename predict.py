# imports 
import argparse
import numpy as np
from PIL import Image
import torch
import json
import torch.nn.functional as F
from torchvision import transforms
from appHelper import gpuSupport, getModelClassifier, getModelSizeFromArch

# app logo
appLogo = """\
                                        _ _      _   
                                       | (_)    | |  
  __ _ _ __  _ __    _ __  _ __ ___  __| |_  ___| |_ 
 / _` | '_ \| '_ \  | '_ \| '__/ _ \/ _` | |/ __| __|
| (_| | |_) | |_) | | |_) | | |  __/ (_| | | (__| |_ 
 \__,_| .__/| .__/  | .__/|_|  \___|\__,_|_|\___|\__|
      | |   | |     | |                              
      |_|   |_|     |_|          
"""
def readInputArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagePath', dest='imagePath', default='./flowers/test/74/image_01254.jpg')    
    parser.add_argument('--loadCategoryFile', dest='loadCategoryFile', default='cat_to_name.json')
    parser.add_argument('--trainedModelName', action='store', default='appCheckpoint.pth')
    parser.add_argument('--topK', dest='topK', type=int, default=5)
    parser.add_argument('--gpu', dest='gpu', type=bool, default=True)
    return parser.parse_args()

# load category file
def loadCategoryFile(fileName):
    with open(fileName) as f:
        catToName = json.load(f)
        return catToName

# process image input
def process_image(image):
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return adjustments(img_pil)

# load model from checkPoint
def loadCheckpointModel(checkpointName):
    print('load checkpoint file :' + str(checkpointName))
    # load data
    checkpoint = torch.load(checkpointName)    
    # get model    
    model = checkpoint['model']
    # set classifier
    model.classifier = checkpoint['classifier']   
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])    
    return model

# convert tensor to array
def convertTensorToStringArray(tensor):
    stringList = []
    for i in tensor.cpu().numpy():
        item = np.char.mod('%d',[i])
        stringList = item[0]
    
    return stringList;

# predict from trained model
def predict(imagePath, model, topk, gpuActiv):
    if gpuActiv:
        model.cuda()
        model.to('cuda')
    else:
        model.cpu()

    img_torch = process_image(imagePath)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    with torch.no_grad():
        output = model.forward(img_torch.cuda())

    probability = F.softmax(output.data,dim=1)
    probs, classes = probability.topk(topk)
    
    return probs.cpu().numpy()[0], convertTensorToStringArray(classes)

# print result from the model
def printResult(probs, classes, catToName):        
    print('\nResults from the model are:')
    print('-----------------------------')
    print(probs)
    print(classes)
    print([catToName[x] for x in classes])
    print('-----------------------------')
    maxIndex = np.argmax(probs)
    label = classes[maxIndex]
    print('Maybe your image is a: ' + catToName[label] + '\n')
    print('\nthx and bye bye...\n')

# main process
def main():
    print(appLogo)
    # read args
    args = readInputArgs()
    # load model
    model = loadCheckpointModel(args.trainedModelName)
    # load category
    catToName = loadCategoryFile(args.loadCategoryFile)
    # check gpu support
    gpuActiv = gpuSupport(args.gpu)
    # predict
    probs, classes = predict(args.imagePath, model, args.topK, gpuActiv)
    # print result
    printResult(probs, classes, catToName)

if __name__ == "__main__":    
    main()