import os
import os.path
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
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
    h,w,c = img.shape
    q = 0
    q_c = 0
    alpha = 0.7
    beta = 4
    gamma = np.log(0.01)
    img = img / 255
    for i in range(c):
        mean_c_i = np.mean(img[:,:,i])
        std_c_i = np.std(img[:,:,i])
        
        q_c = ((alpha * beta * (mean_c_i - (mean_c_i**2))) + ((1- alpha)*(1-np.exp(gamma * std_c_i))))
        q = q + q_c
    
    
    q = q/3
    
    return q

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def create_directories(base_dir, classes, subset):
    for z in classes:
        img_dir = os.path.join(base_dir, subset, z)
        lbp_dir = os.path.join(base_dir, "final_patch_eswa_" + subset, z)
        
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
        lbp_dir = os.path.join(base_dir, "final_patch_eswa_" + subset, z)

        data_path = os.path.join(img_dir,'*g')	
        files = glob.glob(data_path)

        data = []

        for f1 in files:
            img = cv2.imread(f1) 
            height, width, channel = img.shape    
            print("Yes (",height,width,") ",f1)
            temp1 = int(height/64)
            temp2 = int(width/64)
            crop_height = temp1*64
            crop_width = temp2*64


            img_crop = center_crop(img,(crop_width,crop_height))
            h,w,c = img_crop.shape
            print(img_crop.shape)

            img_lbp = np.zeros((64,64,3), np.uint8)

            i=0
            j=0
            k=0
            s = 0
            quality_score = []
            while ((i+64) <= h):
                j=0
                while ((j+64)<= w):

                    img_lbp[:, :, :] = img_crop[i:i+64,j:j+64,:]

                    quality_score.append(isSaturated(img_lbp))

                    filename = str(k)+"_"+os.path.basename(f1)
                    img_dir =  lbp_dir+filename
                    k=k+1
                    j=j+64
                i=i+64
            quality_score = np.sort(quality_score)[::-1]


            len_q = len(quality_score)
            if(len_q<256):
                limit = quality_score[len_q-1]
            else:
                limit = quality_score[255]

            print(limit)

            i=0
            j=0
            k=0

            while ((i+64) <= h):
                j=0
                while ((j+64)<= w):

                    img_lbp[:, :, :] = img_crop[i:i+64,j:j+64,:]

                    if(isSaturated(img_lbp)>=limit):
                        filename = Path(f1).stem+'_'+str(k)+'.jpg'
                        img_dir =  lbp_dir + '/' + filename
                        cv2.imwrite(img_dir, img_lbp)
                        k=k+1
                    j=j+64
                i=i+64
            print(k)

          
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <base_directory>")
        sys.exit(1)

    base_directory = sys.argv[1]

    main(base_directory)