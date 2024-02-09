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
    
    
   def SRM(self):
    
        imgs = self.img_rgb
        
        filter2 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：egde5*5
        filter1 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0,-2, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 12
        filter2 = np.asarray(filter2, dtype=float) / 4
        filter3 = np.asarray(filter3, dtype=float) / 2

        filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]# (3,3,5,5)

        filters = torch.FloatTensor(filters)  
         # (3,3,5,5)
        imgs = np.array(imgs, dtype=float)  # (375,500,3)
        w,h,c = imgs.shape
        imgs = imgs.reshape(1,w,h,c)
        imgs = np.einsum('klij->kjli', imgs)

        
        input = torch.tensor(imgs, dtype=torch.float32)


        op1 = F.conv2d(input, filters, stride=1, padding=2)
        #print('op1\'s shape', op1.shape)

        
        op1= op1.reshape(c,w,h)
    
        self.img_noise = op1

   def __getitem__(self, index):
       path1, target1 = self.imgs1[index]  
       filename = Path(path1).stem 
       img1 = self.loader(os.path.join(self.root1, path1))  
    
       self.img_rgb = img1
       
       
       self.SRM()

       if self.transform_1 is not None:
           img1 = self.transform_1(self.img_rgb)
        
       if self.target_transform is not None:
           target1 = self.target_transform(target)

       return img1,self.img_noise, target1,filename

   def __len__(self):
       return len(self.imgs1)
     
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.resnet1 = models.resnet50(pretrained=True)
        modules1 = list(self.resnet1.children())[:-1]  # delete the last fc layer.
        self.resnet1 = nn.Sequential(*modules1)

        self.resnet2 = models.resnet50(pretrained=True)
        modules2 = list(self.resnet2.children())[:-1]
        self.resnet2 = nn.Sequential(*modules2)

        self.fc1 = nn.Linear(2048 * 2, 2048)
        self.fc2 = nn.Linear(2048, 30)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.resnet1(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.resnet2(x2)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        out_fc = x
        output = self.logsoftmax(x)

        return output, out_fc

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--data', required=True, help='Data directory path')
    parser.add_argument('--model', required=True, help='Model path')
    args = parser.parse_args()

    data_dir = args.data
    model_path = args.model

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    data_transforms = transforms.Compose([
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

    # Check if Results folder exists
    results_folder = "Results"
    if os.path.exists(results_folder):
        # If it exists, delete the folder and its content
        print("Deleting existing Results folder...")
        for file in os.listdir(results_folder):
            file_path = os.path.join(results_folder, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    # Create Results folder
    os.makedirs(results_folder, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (imgs1, imgs2, labels1, patch_filename) in enumerate(test_loader):
            c = labels1
            d = c.cpu().numpy()[0]

            if(d != a):
                print("Yes_Class", d)
                a = d
                z = d
                file_class = os.path.join(results_folder, f"Test_Class_{z}.csv")

                with open(file_class, 'a+', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Batch_Id", "Patch_Filename", "True Class", "Predicted Class", "Probability of Predicted Class"])

            img_org, mat_img, target = imgs1.to(device), imgs2.to(device), labels1.to(device)
            output, fc_feature = model(img_org, mat_img)

            output = F.softmax(fc_feature, dim=1)

            actual = target
            _, predicted = torch.max(output.data, 1)

            y_true = actual.cpu().numpy()[0]
            y_pred = predicted.cpu().numpy()[0]

            prob_y_pred = output[0][y_pred]
            prob_y_pred = prob_y_pred.cpu().numpy()
            prob_y_pred = np.around(prob_y_pred, decimals=2)

            if(batch_idx % 1000 == 0):
                print(batch_idx, patch_filename, y_true, y_pred, prob_y_pred)

            with open(file_class, 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([batch_idx, patch_filename, y_true, y_pred, prob_y_pred])
                
                
    csv_dir = img_dir = "Results/"
    data_path = os.path.join(img_dir,'*csv')
    files = glob.glob(data_path)
    
    with open(os.path.join(csv_dir + 'Image_Level_Results.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Class Label","Number of Images",\
                         "Correct Predicted Images",\
                         "Number of Patches",\
                         "Total Patches(Correct Classified Images)",\
                         "Correct Predicted Patches(Correct Classified Images)",\
                         "Precetange Votes Per Image(Only Correct Images)",\
                         "Average Softmax Probability of Correct Patch(Only Correct Images)"])
        
    for f in files:
      df = pd.read_csv(f)
      data = df.sort_values(by=['Patch_Filename'])
      classname = Path(f).stem
      classname = int(classname.split("_")[2])
      a="a"

      true_image_class = classname
      total_images_perclass = 0
      correct_images_perclass = 0
      total_patches_perclass = 0

      total_class_votes =0
      total_class_patches =0
      prob_avg_correct_patch = 0.0
      votes = 0

      arr_pred_patches = []
      arr_pred_patches_prob = []
      total_correc_img_patches = 0


      for ind in data.index:
              filename = df['Patch_Filename'][ind]
              pred_patch_class = df['Predicted Class'][ind]
              pred_patch_prob = df['Probability of Predicted Class'][ind]
              
              filename = filename.split("_")
              file_length = len(filename)
              initial_filename = filename[:-1]
              patch_name = filename[file_length-1]
              #print(initial_filename)
              #print(patch_name)

              if(a!= initial_filename and a=="a"):
                  a = initial_filename



              if(a!=initial_filename and a!="a"):
                  total_images_perclass = total_images_perclass + 1

                  counts = np.bincount(arr_pred_patches)
                  pred_image_class = np.argmax(counts)
                  votes = np.count_nonzero(arr_pred_patches==pred_image_class)
                  s=0
                  
                  if(pred_image_class == true_image_class):
                      correct_images_perclass = correct_images_perclass + 1
                      total_class_votes = total_class_votes + votes
                      total_class_patches = total_class_patches + len(arr_pred_patches)
                      
                      total_correc_img_patches = total_correc_img_patches + len(arr_pred_patches)

                      z = np.where(np.array(arr_pred_patches)==true_image_class,1,0)
                      for i in range(0,len(arr_pred_patches)):
                          if(z[i]==1):
                              s= s+1
                              prob_avg_correct_patch = prob_avg_correct_patch + arr_pred_patches_prob[i]
              
                  arr_pred_patches = []
                  arr_pred_patches_prob = []
                  a = initial_filename



              if(a==initial_filename):
                  total_patches_perclass = total_patches_perclass + 1
                  arr_pred_patches.append(pred_patch_class)
                  arr_pred_patches_prob.append(pred_patch_prob)



      total_images_perclass = total_images_perclass + 1

      counts = np.bincount(arr_pred_patches)
      pred_image_class = np.argmax(counts)
      votes = np.count_nonzero(arr_pred_patches==pred_image_class)

      if(pred_image_class == true_image_class):
          correct_images_perclass = correct_images_perclass + 1
          total_class_votes = total_class_votes + votes
          total_class_patches = total_class_patches + len(arr_pred_patches)
          
          total_correc_img_patches = total_correc_img_patches + len(arr_pred_patches)

          z = np.where(np.array(arr_pred_patches)==true_image_class,1,0)
          for i in range(0,len(arr_pred_patches)):
              if(z[i]==1):
                  prob_avg_correct_patch = prob_avg_correct_patch + arr_pred_patches_prob[i]

      arr_pred_patches = []
      arr_pred_patches_prob = []
      
      if(correct_images_perclass!=0):
          prob_avg_correct_patch = prob_avg_correct_patch / total_class_votes
          prob_avg_correct_patch = np.around(prob_avg_correct_patch,decimals=2)
          avg_vote_perclass = np.around((total_class_votes*100) / total_correc_img_patches,decimals=2)
          print(total_images_perclass,total_patches_perclass,correct_images_perclass,total_class_votes,avg_vote_perclass,prob_avg_correct_patch)  

          with open(os.path.join(csv_dir + 'Image_Level_Results.csv'), 'a+', newline='') as file:
              writer = csv.writer(file)
              writer.writerow([classname,total_images_perclass,correct_images_perclass,total_patches_perclass,\
                              total_correc_img_patches,total_class_votes,\
                              avg_vote_perclass,\
                              prob_avg_correct_patch])
      else:
          with open(os.path.join(csv_dir + 'Image_Level_Results.csv'), 'a+', newline='') as file:
              writer = csv.writer(file)
              writer.writerow([classname,total_images_perclass,correct_images_perclass,total_patches_perclass,\
                              total_correc_img_patches,total_class_votes,\
                              avg_vote_perclass,\
                              prob_avg_correct_patch])
              
              
    df = pd.read_csv(os.path.join(csv_dir + 'Image_Level_Results.csv'))

    ILA = sum(df['Correct Predicted Images']) / sum(df['Number of Images'])
    
    print(f"Image Level Accuracy: {ILA}")
