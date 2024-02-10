import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import os.path
import os
import glob
from pathlib import Path

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    crop_width = dim[0]
    crop_height = dim[1]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def isSaturated(img):
    
    img = img / 255
    
    
    std_c_1 = np.std(img[:,:,0])
    std_c_2 = np.std(img[:,:,1])
    std_c_3 = np.std(img[:,:,2])
    
    
    if(std_c_1 < 0.005 and std_c_2 < 0.005 and std_c_3 < 0.005 ):
        return True
    else:
        return False
    

def isNonHomogeneous(img):
    
    img = img / 255
    
    
    std_c_1 = np.std(img[:,:,0])
    std_c_2 = np.std(img[:,:,1])
    std_c_3 = np.std(img[:,:,2])
    
    
    if(std_c_1 > 0.02 or std_c_2 > 0.02 and std_c_3 > 0.02 ):
        return True
    else:
        return False
    

def isHomogeneous(img):
    
    img = img / 255
    
    std_c_1 = np.std(img[:,:,0])
    std_c_2 = np.std(img[:,:,1])
    std_c_3 = np.std(img[:,:,2])
    
    if((std_c_1 >= 0.005 and std_c_2 >= 0.005 and std_c_3 >= 0.005) and  (std_c_1 <= 0.02 and std_c_2 <= 0.02 and std_c_3 <= 0.02)):
        return True
    else:
        return False
    
def find_classes(dir):
   classes = os.listdir(dir)
   classes.sort()
   class_to_idx = {classes[i]: i for i in range(len(classes))}
   return classes, class_to_idx


def create_directories(base_dir, classes, subset):
    for z in classes:
        img_dir = os.path.join(base_dir, subset, z)
        lbp_dir = os.path.join(base_dir, "final_patch_benna_" + subset, z)
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbp_dir, exist_ok=True)

def main(base_dir):
    # Similar
    classes_similar_train, _ = find_classes(os.path.join(base_dir, "similar/train"))
    create_directories(base_dir, classes_similar_train, "similar/train")
    process_images(base_dir, classes_similar_train, "similar/train")
    
    classes_similar_test, _ = find_classes(os.path.join(base_dir, "similar/test"))
    create_directories(base_dir, classes_similar_test, "similar/test")
    process_images(base_dir, classes_similar_test, "similar/test")

    # Random
    classes_random_train, _ = find_classes(os.path.join(base_dir, "random/train"))
    create_directories(base_dir, classes_random_train, "random/train")
    process_images(base_dir, classes_random_train, "random/train")
    
    classes_random_test, _ = find_classes(os.path.join(base_dir, "random/test"))
    create_directories(base_dir, classes_random_test, "random/test")
    process_images(base_dir, classes_random_test, "random/test")

    # Merged
    classes_merged_train, _ = find_classes(os.path.join(base_dir, "merged/train"))
    create_directories(base_dir, classes_merged_train, "merged/train")
    process_images(base_dir, classes_merged_train, "merged/train")
    
    classes_merged_test, _ = find_classes(os.path.join(base_dir, "merged/test"))
    create_directories(base_dir, classes_merged_test, "merged/test")
    process_images(base_dir, classes_merged_test, "merged/test")


def process_images(base_dir, classes, subset):
  for z in classes:
      img_dir = os.path.join(base_dir, subset, z)
      lbp_dir = os.path.join(base_dir, "final_patch_benna_" + subset, z)


      data_path = os.path.join(img_dir,'*g')
      files = glob.glob(data_path)
      #print(files)
      data = []
      #print(img_dir)
      #print(lbp_dir)

      for f1 in files:

              img = cv2.imread(f1) 
              height, width, channel = img.shape    
              print("Yes (",height,width,") ",f1)
              temp1 = int(height/128)
              temp2 = int(width/128)
              crop_height = temp1*128
              crop_width = temp2*128
              
              img_crop = center_crop(img,(crop_width,crop_height))
              h,w,c = img_crop.shape
              
              img_lbp = np.zeros((128,128,3), np.uint8)
              img_lbp_double = np.zeros((128,128,3))
              
              i = 0
              k = 0
              while ((i+128) <= h):
                  j=0
                  while ((j+128)<= w and k<400):
                      img_lbp[:, :, :] = img_crop[i:i+128,j:j+128,:]
                      
                      if(isHomogeneous(img_lbp)):
                          filename = Path(f1).stem+'_'+str(k)+'.jpg'
                          img_dir =  lbp_dir + '/' + filename
                          cv2.imwrite(img_dir, img_lbp)
                          k=k+1
                      j=j+32
                  i=i+32
              
              print(k)
              
              if(k<400):
                  i = 0
                  while ((i+128) <= h):
                      j=0
                      while ((j+128)<= w and k<400):
                          img_lbp[:, :, :] = img_crop[i:i+128,j:j+128,:]
                          img_lbp_double[:, :, :] = img_crop[i:i+128,j:j+128,:]

                          if(isSaturated(img_lbp) and (isHomogeneous(img_lbp)==False)):
                              filename = Path(f1).stem+'_'+str(k)+'.jpg'
                              img_dir =  lbp_dir + '/' + filename
                              cv2.imwrite(img_dir, img_lbp)
                              k=k+1
                          j=j+32
                      i=i+32
                  
              if(k<400):
                  i = 0
                  while ((i+128) <= h):
                      j=0
                      while ((j+128)<= w and k<400):
                          img_lbp[:, :, :] = img_crop[i:i+128,j:j+128,:]

                          if(isNonHomogeneous(img_lbp) and (isHomogeneous(img_lbp)==False) and (isSaturated(img_lbp)==False)):
                              filename = Path(f1).stem+'_'+str(k)+'.jpg'
                              img_dir =  lbp_dir + '/' + filename
                              cv2.imwrite(img_dir, img_lbp)
                              k=k+1
                          j=j+32
                      i=i+32
              
              print(k)
              


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <base_directory>")
        sys.exit(1)

    base_directory = sys.argv[1]

    main(base_directory)
        