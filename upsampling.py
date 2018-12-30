#if need
# !pip install pillow
# !pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely


import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import Augmentor as Aug
import matplotlib.pyplot as plt

!ls /home/phan/HPA
root_path = '/home/phan/HPA/'
Ex_im_path = '/home/phan/HPA/Ex/train/' #folder which stores External train image
Raw_im_path = '/home/phan/HPA/Raw/train/' #folder which stores Kaggle train image

upsampling_path = '/home/phan/HPA/Min/' #folder store minority csv file for both Kaggle and external data
upsampling_Rw_csv = upsampling_path + "Raw_upspl.csv" #save upsampling csv file for Kaggle data
upsampling_Ex_csv = upsampling_path + "Ex_upspl.csv" #save upsampling csv file for External data

Ex_usp_path = '/home/phan/HPA/Min/Ex/' # folder save Augmented minority file belonging to External data
Rw_usp_path = '/home/phan/HPA/Min/Raw/' # folder save Augmented minority file belonging to Kaggle rw data

# !rm -rf /home/phan/HPA/Min/Ex/* 
# !mkdir /home/phan/HPA/Min/Ex

# !rm -rf /home/phan/HPA/Min/Raw/*
# !mkdir /home/phan/HPA/Min/Ex 

# !ls Rw_usp_path  

# print (Rw_usp_path, Ex_usp_path)

#==================================================================================================================
#upsampling for Raw data 
#load data
path2 = upsampling_path + "Rw_minority.csv"
df2 = pd.read_csv(path2)
print(len(df2))

colors = ['red','green','blue','yellow']
aug_labels = ['flipv','fliph','rot15','rot30','rot45'] # use 5 aug mechanism, remove any if don't need 
aug_labels = ['flipv'] #for test omly
# Read image
index = 0
new_data_to_csv = []
for i in df2['Id'][:5]: # [:5] for test only first 5 samples
    print('image: ', i)
    Id = i
    Target = df2.Target[index]
    index += 1
    print('labels: ', Target)
    for color in colors:        
        print(color)
        image_path = Raw_im_path + i + "_" + color + ".png" # need add color 
        print(image_path)
        img = Image.open(image_path) #load image to prepare for augmentation
        img = np.array(img)
        plt.imshow(img)   # img.show()
        plt.show()     
        for aug in aug_labels:        
            img_new_name = i + "_" + color  + "_" + aug + ".png"            
            if aug == 'flipv': # flip vertical
                print('flipv')                
                flipv = np.fliplr(img)
                imfsv = Image.fromarray(flipv) 
                imfsv.save(Ex_usp_path + img_new_name)
               
            elif aug == 'fliph':#flip horizontal
                print('fliph')
                flipv = np.fliplr(img)
                imfsv = Image.fromarray(flipv) 
                imfsv.save(Ex_usp_path + img_new_name)
                
            elif aug == 'rot15': #rotate 15 degree
                print('rot15')
                lipv = np.fliplr(img)
                imfsv = Image.fromarray(flipv) 
                imfsv.save(Ex_usp_path + img_new_name)
                
            elif aug == 'rot30': #rotate 15 degree
                print('rot30')
                flipv = np.fliplr(img)
                imfsv = Image.fromarray(flipv) 
                imfsv.save(Ex_usp_path + img_new_name)
                
            elif aug == 'rot45': #rotate 15 degree
                print('rot45')
                flipv = np.fliplr(img)
                imfsv = Image.fromarray(flipv) 
                imfsv.save(Ex_usp_path + img_new_name)
                
            else:
                print('check image: ', i)
                pass
            new_information = [img_new_name, Target] #new name with old labels
            new_data_to_csv.append(new_information) 
            print(new_information)

#write upsampling data [Id, Target] to file csv
upsample_data = pd.DataFrame(data=new_data_to_csv, columns=['Id','Target'])
print(upsample_data.shape)
upsample_data.to_csv(upsampling_Rw_csv, index=False)



#==================================================================================================================
#upsampling External data 
#load data
path1 = upsampling_path + "Ex_minority.csv"
df1 = pd.read_csv(path1)
print(len(df1))

colors = ['red','green','blue','yellow']
# aug_labels = ['flipv','fliph','rot15','rot30','rot45']
aug_labels = ['flipv']
# Read image
index = 0
new_data_to_csv = []
for i in df1['Id'][:5]: # [:5] for test only first 5 samples
    print('image: ', i)
    Id = i
    Target = df1.Target[index]
    index += 1
    print('labels: ', Target)
    for color in colors:        
        print(color)
        read_image_path = Ex_im_path + i + "_" + color + ".png" # need link to folder && add color 
        print(image_path)
        img = Image.open(image_path) #load image to prepare for augmentation
        img = np.array(img)
        plt.imshow(img)   # img.show()
        plt.show()     
        for aug in aug_labels:        
            img_new_name = i + "_" + color  + "_" + aug + ".png"            
            if aug == 'flipv':
                print('flipv')                
                flipv = np.fliplr(img)
                imfsv = Image.fromarray(flipv) 
                imfsv.save(Rw_usp_path + img_new_name)
               
            elif aug == 'fliph':
                print('fliph')
                flipv = np.fliplr(img)
                imfsv = Image.fromarray(flipv) 
                imfsv.save(Rw_usp_path + img_new_name)
                
            elif aug == 'rot15':
                print('rot15')
                lipv = np.fliplr(img)
                imfsv = Image.fromarray(flipv) 
                imfsv.save(Rw_usp_path + img_new_name)
                
            elif aug == 'rot30':
                print('rot30')
                flipv = np.fliplr(img)
                imfsv = Image.fromarray(flipv) 
                imfsv.save(Rw_usp_path + img_new_name)
                
            elif aug == 'rot45':
                print('rot45')
                flipv = np.fliplr(img)
                imfsv = Image.fromarray(flipv) 
                imfsv.save(Rw_usp_path + img_new_name)
                
            else:
                print('check image: ', i)
                pass
            new_information = [img_new_name, Target] #new name with old labels
            new_data_to_csv.append(new_information) 
            print(new_information)

#write upsampling data [Id, Target] to file csv
upsample_data = pd.DataFrame(data=new_data_to_csv, columns=['Id','Target'])
print(upsample_data.shape)
upsample_data.to_csv(upsampling_Ex_csv, index=False)


#==================================================================================================================
#Load data function
def load_train_data():
    # minority set inside external data
    path1 = upsampling_Ex_csv 

    # minority set inside kaggle raw data
    path2 = upsampling_Ex_csv 

    # kaggle raw data
    path3 = root_path + "Ex/train.csv"
    
    # external data
    path4 = root_path + "Raw/train.csv"


    # load data
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    df4 = pd.read_csv(path4)

    df = pd.concat([df3, df4, df1, df2])

    return df