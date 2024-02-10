from __future__ import print_function
from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision
import glob
import cv2
import matplotlib.pyplot as plt
import csv
from torchvision import models
import time
import os
import copy
import scipy.io
import pandas as pd
import numpy as np
from pathlib import Path
import torch.utils.data as data
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
   '.jpg', '.JPG', '.jpeg', '.JPEG',
   '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.mat',
]


def is_image_file(filename):
   return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
   classes = os.listdir(dir)
   classes.sort()
   class_to_idx = {classes[i]: i for i in range(len(classes))}
   return classes, class_to_idx


def make_dataset(dir, class_to_idx):
   images = []
   for target in os.listdir(dir):
       d = os.path.join(dir, target)
       if not os.path.isdir(d):
           continue

       for filename in os.listdir(d):
           if is_image_file(filename):
               path = '{0}/{1}'.format(target, filename)
               #print(path)
               item = (path, class_to_idx[target])
               images.append(item)

   return images

def default_loader(path):
   return Image.open(path).convert('RGB')

def mat_loader(path):
   return scipy.io.loadmat(path1)
  
class ImageFolderLoader(data.Dataset):
   def __init__(self, root1,transform_1=None,
                target_transform=None,
                loader=default_loader):
       classes1, class_to_idx1 = find_classes(root1)
       
       imgs1 = make_dataset(root1, class_to_idx1)
      

       self.root1 = root1
       self.imgs1 = imgs1
       self.classes1 = classes1
       self.class_to_idx1 = class_to_idx1
       self.transform_1 = transform_1
       self.target_transform = target_transform
       self.loader = loader
        
       self.img_noise = None
       self.img_rgb = None
    
    
   def __getitem__(self, index):
    

       path1, target1 = self.imgs1[index]    
       filename = Path(path1).stem 
       img1 = self.loader(os.path.join(self.root1, path1))  
    
       self.img_rgb = img1
       

       if self.transform_1 is not None:
           img1 = self.transform_1(self.img_rgb)
        
       if self.target_transform is not None:
           target1 = self.target_transform(target)
            
       
       return img1,target1

   def __len__(self):
       return len(self.imgs1)
     
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        resnet1 = models.resnet50(pretrained=True)
        modules1 = list(resnet1.children())[:-1]      # delete the last fc layer.
        self.resnet1 = nn.Sequential(*modules1)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048,30)
        
        
        self.logsoftmax = nn.LogSoftmax(dim=1) 

    def forward(self, x):
        
        x = self.resnet1(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
       
        out_fc = x
        output = self.logsoftmax(x)
        
        return output, out_fc
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Data directory path')
    parser.add_argument('--model', required=True, help='Model path')
    args = parser.parse_args()

    data_dir = args.data
    model_path = args.model

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    data_transforms = transforms.Compose([
      transforms.Resize((256,256)),
      transforms.ToTensor()
    ])
    
    classes1, class_to_idx1 = find_classes(data_dir)
    imgs1 = make_dataset(data_dir, class_to_idx1)
    print(class_to_idx1,len(imgs1))
       
    imgs1 = make_dataset(data_dir, class_to_idx1)

    batchsize = 1

    val_dataset = ImageFolderLoader(
        data_dir,
        transform_1=data_transforms
    )

    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batchsize,
        shuffle=False, num_workers=4
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0')
    model = Net().to(device)

    model = torch.load(model_path)
    model.eval()

    a = -1
    criterion = nn.NLLLoss()

        
    test_loss = 0
    total_loss = 0
    train_loss = 0
    correct = 0
    test_epoch = 1
    total_correct_test = 0
    total_correct_train = 0
    total_correct = 0
    epoch = 0

    with torch.no_grad():
        for batch_idx, (imgs1,labels1) in enumerate(test_loader):
            
            img_org,target = imgs1.to(device),labels1.to(device)
            output, fc_feature = model(img_org)
            loss = criterion(output, target)

            test_loss = test_loss + loss.item()  # sum up batch loss

            _, predicted = torch.max(output.data, 1)
            total = float(len(target))
            correct = (predicted == target).sum()
            total_correct_test = total_correct_test + np.float64(correct.cpu().numpy())


            if batch_idx % 200 == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(img_org), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))

            if batch_idx % 200 == 0:
                accuracy = 100.0 * float(correct) / total
                print("Accuracy===",accuracy)


    alpha = (len(test_loader.dataset))/ batchsize
    print(test_loss)
    test_loss /= alpha

    acc= 100. * total_correct_test / len(test_loader.dataset)
    np_acc=np.around((100* total_correct_test / len(test_loader.dataset)),decimals=2)


    print('\nImage Level Accuracy: ({:.0f}%)\n'.format(acc))
