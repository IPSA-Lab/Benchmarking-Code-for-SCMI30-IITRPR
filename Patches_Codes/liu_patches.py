import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import os.path
import glob
from pathlib import Path
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import KDTree


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
        mean_c_i = mean_c_i /255
        std_c_i = np.std(img[:,:,i])
        std_c_i = std_c_i/255
        
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
        lbp_dir = os.path.join(base_dir, "final_patch_liu_" + subset, z)
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbp_dir, exist_ok=True)

def main(base_dir):
    # Similar
    classes_similar_train, _ = find_classes(os.path.join(base_dir, "similar/train"))
    create_directories(base_dir, classes_similar_train, "similar/train")
    process_images(base_dir, classes_similar_train, "similar/train")
    # process_images_2(base_dir, classes_similar_train, "similar/train")

    
    classes_similar_test, _ = find_classes(os.path.join(base_dir, "similar/test"))
    create_directories(base_dir, classes_similar_test, "similar/test")
    process_images(base_dir, classes_similar_test, "similar/test")
    # process_images_2(base_dir, classes_similar_test, "similar/test")


    # Random
    classes_random_train, _ = find_classes(os.path.join(base_dir, "random/train"))
    create_directories(base_dir, classes_random_train, "random/train")
    process_images(base_dir, classes_random_train, "random/train")
    # process_images_2(base_dir, classes_random_train, "random/train")

    
    classes_random_test, _ = find_classes(os.path.join(base_dir, "random/test"))
    create_directories(base_dir, classes_random_test, "random/test")
    process_images(base_dir, classes_random_test, "random/test")
    # process_images_2(base_dir, classes_random_test, "random/test")


    # Merged
    classes_merged_train, _ = find_classes(os.path.join(base_dir, "merged/train"))
    create_directories(base_dir, classes_merged_train, "merged/train")
    process_images(base_dir, classes_merged_train, "merged/train")
    # process_images_2(base_dir, classes_merged_train, "merged/train")

    
    classes_merged_test, _ = find_classes(os.path.join(base_dir, "merged/test"))
    create_directories(base_dir, classes_merged_test, "merged/test")
    process_images(base_dir, classes_merged_test, "merged/test")
    # process_images_2(base_dir, classes_merged_test, "merged/test")


def process_images(base_dir, classes, subset):
  
  mean_image = np.zeros((64,64,3), np.float64)
  total = 0
  mean_image_2 = mean_image /total
  print(np.sum(mean_image_2))
  mean_image_2 = 0
  for z in classes:
      img_dir = os.path.join(base_dir, subset, z)
      lbp_dir = os.path.join(base_dir, "final_patch_liu_" + subset, z)


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
                      img_dir =  lbp_dir + '/' +filename
                      #print(img_dir,i,j)
                      #cv2.imwrite(img_dir, img_lbp)
                      #print(img_crop[i:i+32,j:j+32,:].shape)
                      #print(i,j)
                      k=k+1
                      j=j+64
                  i=i+64
              quality_score = np.sort(quality_score)[::-1]
              
              
              len_q = len(quality_score)
              if(len_q<64):
                  limit = quality_score[len_q-1]
              else:
                  limit = quality_score[63]
                  
              
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
                          
                          #print(img_dir,i,j)
                          cv2.imwrite(img_dir, img_lbp)
                          mean_image = mean_image + img_lbp.astype(np.float64)
                          total +=1 
                          #print(img_crop[i:i+32,j:j+32,:].shape)
                          #print(i,j)
                          k=k+1
                          if(k==64):
                              i = h + 1
                              j = w + 1
                      j=j+64
                  i=i+64
              print(k)


# def process_images_2(base_dir, classes, subset):

  for z in classes:
      img_dir = os.path.join(base_dir, subset, z)
      lbp_dir = os.path.join(base_dir, "final_patch_liu_" + subset, z)
      
      
      data_path = os.path.join(img_dir,'*g')
      files = glob.glob(data_path)
      #print(files)
      data = []
      #print(img_dir)
      #print(lbp_dir)

      for f1 in files:
          print(f1)
          
          
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

          all_mean_std = []

          mean =0
          std =0
          while ((i+64) <= h):
              j=0
              while ((j+64)<= w):

                  img_lbp[:, :, :] = img_crop[i:i+64,j:j+64,:]

                  mean=0.0
                  std =0.0
                  k = k +1
                  for z in range(3):
                      mean_c_i = np.mean(img_lbp[:,:,z])
                      mean = mean + (mean_c_i/255)
                      std_c_i = np.std(img_lbp[:,:,z])
                      std = std + (std_c_i/255)


                  j = j + 64
                  mean = mean /3
                  std = std /3
                  all_mean_std.append([mean,std])



              i = i  + 64


          all_mean_std_numpy = np.array(all_mean_std)

          kmeans = KMeans(n_clusters=16, random_state=0).fit(all_mean_std_numpy)
          centroids = kmeans.cluster_centers_

          selected_patches=np.zeros((64,2))
          print(len(centroids))

          q=0
          for p in range(16):

              temp = all_mean_std_numpy
              temp = KDTree(temp)
              distances, ids = temp.query(centroids[p].reshape(1,2), 4)

              selected_patches[q][:] = all_mean_std_numpy[ids[0][0]][:]
              selected_patches[q+1][:] = all_mean_std_numpy[ids[0][1]][:]
              selected_patches[q+2][:] = all_mean_std_numpy[ids[0][2]][:]
              selected_patches[q+3][:] = all_mean_std_numpy[ids[0][3]][:]

              q = q + 4   


          i=0
          j=0
          k=0
          s = 0
          per_image_patch = 0
          

          temp_selected_patches = selected_patches * 10000000
          temp_selected_patches = np.trunc(temp_selected_patches)

          while ((i+64) <= h):
              j=0
              while ((j+64)<= w):

                  img_lbp[:, :, :] = img_crop[i:i+64,j:j+64,:]

                  mean=0.0
                  std =0.0
                  k = k +1
                  for z in range(3):
                      mean_c_i = np.mean(img_lbp[:,:,z])
                      mean = mean + (mean_c_i/255)
                      std_c_i = np.std(img_lbp[:,:,z])
                      std = std + (std_c_i/255)



                  mean = mean /3
                  std = std /3

                  temp_mean_std = np.array([[mean,std]])


                  temp_mean_std = temp_mean_std * 10000000
                  temp_mean_std = np.trunc(temp_mean_std)

                  for t in range(0,len(selected_patches)):
                      s_p = temp_selected_patches[t].reshape(1,2)
                      #print(t)
                      if((temp_mean_std == s_p).all()):
                          #print(True)
                          filename = Path(f1).stem+'_'+str(per_image_patch+64)+'.jpg'
                          img_dir =  lbp_dir + '/' +filename
                          cv2.imwrite(img_dir, img_lbp)   
      
                          per_image_patch = per_image_patch + 1
                          
                          if(per_image_patch==64):
                              j = w + 1
                              i = h + 1
                          
                          
                  j= j+64
              i=i+64


          print(per_image_patch)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <base_directory>")
        sys.exit(1)

    base_directory = sys.argv[1]

    main(base_directory)
    
    


            
        