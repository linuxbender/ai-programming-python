
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import transforms, datasets, models


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# DONE: Define your transforms for the training, validation, and testing sets
# train: random scaling, cropping, and flipping. 
# resized to 224x224 pixels
# means, it's [0.485, 0.456, 0.406]
# standard deviations [0.229, 0.224, 0.225]
# Based on transfer learning lesson solution
BATCH_SIZE = 64
SHUFFLE = True
NUMBER_224 = 224
NUMBER_256 = 256

data_transforms = {
    'train' : transforms.Compose([transforms.RandomResizedCrop(NUMBER_224),transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                                    transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                                                            
    'valid' : transforms.Compose([transforms.Resize(NUMBER_256),transforms.CenterCrop(NUMBER_224),transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),

    'test' : transforms.Compose([transforms.Resize(NUMBER_256),transforms.CenterCrop(NUMBER_224),transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
}
# DONE: Load the datasets with ImageFolder
image_datasets = {
    'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
}
# DONE: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=SHUFFLE),
    'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=SHUFFLE),
    'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE, shuffle=SHUFFLE)   
}


# In[4]:


# check data load
# check gpu 
# torch.cuda.is_available()
print("List size: "+ str(len(dataloaders)))
print("train image size: " + str(len(dataloaders['train'])))
print("test image size: " + str(len(dataloaders['test'])))
print("valid image size: " + str(len(dataloaders['valid'])))


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[5]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.
# 
# Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

# In[6]:


# DONE: Build and train your network
# https://pytorch.org/docs/stable/torchvision/models.html
model = models.vgg13(pretrained=True) # 25088


# In[11]:


# Freeze parameters on the pre-trained model, we will build our own classifier
for param in model.parameters():
    param.requires_grad = False


# In[8]:


def getModelClassifier():
    input_size = 25088
    hidden_sizes = [128, 96]
    output_size = 102
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('drop1', nn.Dropout(p=0.5)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('drop2', nn.Dropout(p=0.4)),
        ('relu2', nn.ReLU()),
        ('fc4', nn.Linear(hidden_sizes[1], output_size)),    
        ('output', nn.LogSoftmax(dim=1))]))
    return classifier
model.classifier = getModelClassifier()


# In[9]:


# criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


# In[10]:


def train(epochs=4):
    
    model.to('cuda')
    print_every = 5
    steps = 0
    loss_show=[]
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputTrain, labelTrain) in enumerate(dataloaders['train']):
            steps += 1            
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
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost),
                       "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0


# In[11]:


train()


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[228]:


# DONE: Do validation on the test set
def validateWithTestSet():   
    model.to('cuda')    
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on the test images is: %d %%' % (100 * correct / total))


# In[227]:


validateWithTestSet()


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[14]:


# DONE: Save the checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu
torch.save({'structure' :'vgg13','fc2':128,'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},'checkpoint.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[15]:


# DONE: Write a function that loads a checkpoint and rebuilds the model
def getCheckPointFromFile():
    checkpoint = torch.load('checkpoint.pth')
    model = models.vgg13(pretrained=True)
    model.classifier = getModelClassifier()  
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model
    
loadModel = getCheckPointFromFile()
print(loadModel)


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[ ]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    return adjustments(img_pil)
    
# DONE: Process a PIL image for use in a PyTorch model
img = (data_dir + '/test/1/image_06752.jpg')
img = process_image(img)
print(img.shape)


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[220]:


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[221]:


def convertTensorToStringArray(tensor):
    stringList = []
    for i in tensor.cpu().numpy():
        item = np.char.mod('%d',[i])
        stringList = item[0]
    
    return stringList;

def predict(imagePath, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # DONE: Implement the code to predict the class from an image file    
    model.to('cuda')
    img_torch = process_image(imagePath)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    with torch.no_grad():
        output = model.forward(img_torch.cuda())

    probability = F.softmax(output.data,dim=1)
    probs, classes = probability.topk(topk)
    
    return probs.cpu().numpy()[0], convertTensorToStringArray(classes)


# In[222]:


imgPath = (test_dir + '/74/image_01254.jpg')

probs, classes = predict(imgPath, model)
print(probs)
print(classes)


# In[223]:


maxIndex = np.argmax(probs)
label = classes[maxIndex]
print('Maybe a: ' + cat_to_name[label])
with  Image.open(imgPath) as image:
    plt.imshow(image)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[214]:


# DONE: Display an image along with the top 5 classes
# Thanks to ai ng project slack community
def displayTopFive(imgPath):
    probs, classes = predict(imgPath, model)
    maxIndex = np.argmax(probs)
    label = classes[maxIndex]

    fig = plt.figure(figsize=(10,10))
    ax1 = plt.subplot2grid((15,9),(0,0),colspan=9, rowspan=9)
    ax2 = plt.subplot2grid((15,9),(9,2),colspan=5, rowspan=5)
    
    image = Image.open(imgPath)
    ax1.axis('off')
    ax1.set_title(cat_to_name[label])
    ax1.imshow(image)
    labels = []
    for cl in classes:
        labels.append(cat_to_name[cl])
    y_pos = np.arange(5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.barh(y_pos, probs, xerr=0, align='center')
    plt.show()


# In[215]:


displayTopFive(test_dir + '/55/image_04740.jpg')


# In[216]:


displayTopFive(test_dir + '/64/image_06099.jpg')


# In[217]:


displayTopFive(test_dir + '/74/image_01254.jpg')

