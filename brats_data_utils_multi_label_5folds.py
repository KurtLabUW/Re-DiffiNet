# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 18:04:51 2023

@author: Kurtlab
"""

from sklearn.model_selection import KFold  
import matplotlib.pyplot as plt
import os
import json
import math
import numpy as np
import torch
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
import pickle
import random
import pandas as pd
def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
def channelchangetomonai(y1):
    y1 = np.where(y1==3, 4,y1) 
    return y1
    
        


def resample_img(
    image: sitk.Image,
    out_spacing = (2.0, 2.0, 2.0),
    out_size = None,
    is_label: bool = False,
    pad_value = 0.,
) -> sitk.Image:
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert our z, y, x convention to SimpleITK's convention
    out_spacing = list(out_spacing)[::-1]

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image

class PretrainDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d  = self.read_data(datalist[i])
                self.cache_data.append(d)




    def read_data(self,data_path):       
        x1, x2, x3, x4, y1,yp = pkload(data_path) 

        y1, yp = channelchangetomonai(y1), channelchangetomonai(yp)
        x1=np.squeeze(x1)
        x2=np.squeeze(x2)
        x3=np.squeeze(x3)
        x4=np.squeeze(x4)
        y1=np.squeeze(y1)
        yp=np.squeeze(yp)


           
        y_3channel = np.stack(((np.zeros(yp.shape)),np.zeros(yp.shape),np.zeros(yp.shape)), axis=0)
        y_3channel[0,:,:,:] = np.where(yp ==1 , 1.,0.)
        y_3channel[1,:,:,:] = np.where(yp ==2 , 1.,0.)
        y_3channel[2,:,:,:] = np.where(yp ==4 , 1.,0.)
        y_3channel[0,:,:,:] = y_3channel[1,:,:,:] + y_3channel[0,:,:,:] + y_3channel[2,:,:,:]
        y_3channel[1,:,:,:] = y_3channel[0,:,:,:] - y_3channel[1,:,:,:] 

        y1_3channel = np.stack(((np.zeros(yp.shape)),np.zeros(yp.shape),np.zeros(yp.shape)), axis=0)
        y1_3channel[0,:,:,:] = np.where(y1 ==1 , 1.,0.)
        y1_3channel[1,:,:,:] = np.where(y1 ==2 , 1.,0.)
        y1_3channel[2,:,:,:] = np.where(y1 ==4 , 1.,0.)
        y1_3channel[0,:,:,:] = y1_3channel[1,:,:,:] + y1_3channel[0,:,:,:] + y1_3channel[2,:,:,:]
        y1_3channel[1,:,:,:] = y1_3channel[0,:,:,:] - y1_3channel[1,:,:,:]
        
        
        
        
        y1_delta=np.squeeze(y_3channel) - np.squeeze(y1_3channel)
        y1_delta= np.where(y1_delta==-1,1,y1_delta)
        

        diffusion_type = 'original' # delta or original
        use_UNetopt = False
        cat_UNetopt = True
        apply_UNetopt = False
        if use_UNetopt: 
            image_data = y_3channel.astype(np.float32)
        if cat_UNetopt:
            image_data =  np.stack((x1,x2,x3,x4,y_3channel[0,],y_3channel[1,],y_3channel[2,]), axis=0).astype(np.float32)
        if apply_UNetopt:
            alpha = 0.8
            image_data = np.stack((np.where(yp >0, x1, (1-alpha)*x1),np.where(yp >0, x2, (1-alpha)*x2),np.where(yp >0, x3, (1-alpha)*x3),np.where(yp >0, x4, (1-alpha)*x4)), axis=0).astype(np.float32)    
        # else:
        #     image_data =  np.stack((x1,x2,x3,x4), axis=0).astype(np.float32)
            
        if 'delta' in diffusion_type: 
            seg_data = y1_delta #y1_3channel, y1
            tumor = y1_3channel
        else:
            seg_data = y1_3channel #y1_3channel, np.expand_dims((y1), axis=0).astype(np.int32)
            tumor = y1_3channel


        
        return {
            "image": image_data,
            "label": seg_data,
            "tumor": tumor,
        } 
    

    # def read_data(self,data_path):
    #     x1, x2, x3, x4, y1, yp = pkload(data_path) 
    #     x1 = np.transpose(x1)
    #     x2 = np.transpose(x2)
    #     x3 = np.transpose(x3)
    #     x4 = np.transpose(x4)
    #     y1 = np.transpose(y1)
    #     yp = np.transpose(yp)
    #     image_data = np.stack((x1,x2,x3,x4), axis=0).astype(np.float32)
    #     seg_data =  [np.expand_dims((y1), axis=0).astype(np.int32), np.expand_dims((y1), axis=0).astype(np.int32)]
        
    #     return {
    #         "image": image_data,
    #         "label": seg_data

    #     } 


    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else :
            try:
                image = self.read_data(self.datalist[i])
            except:
                with open("./bugs.txt", "w+") as f:
                    f.write(f"bug in dataloader，{self.datalist[i]}\n")
                if i != len(self.datalist)-1:
                    return self.__getitem__(i+1)
                else :
                    return self.__getitem__(i-1)
        if self.transform is not None :
            image = self.transform(image)
        
        return image

    def __len__(self):
        return len(self.datalist)

def get_kfold_data(data_paths, n_splits, shuffle=False):
    X = np.arange(len(data_paths))
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)  
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            fold_train.append(data_paths[i])
        for j in b:
            fold_val.append(data_paths[j])
        return_res.append({"train_data": fold_train, "val_data": fold_val})

    return return_res

class Args:
    def __init__(self) -> None:
        self.workers=8
        self.fold=0
        self.batch_size=2

def get_loader_brats(data_dir, batch_size=1, fold=0, num_workers=8):

    #all_dirs = os.listdir(data_dir)
    
    #
    all_dirs_train = os.listdir(data_dir+'trainning')
    all_dirs_validation = os.listdir(data_dir+'validation')
    
    
    
    
    all_paths = [os.path.join(data_dir+'trainning', d) for d in all_dirs_train]
    val_paths = [os.path.join(data_dir+'validation', d) for d in all_dirs_validation]
   
    all_paths = all_paths + val_paths 

       
    size = len(all_paths)
    
    np.random.seed(42)
    np.random.shuffle(all_paths)
    
#    train_size = int(0.7 * size)
#    val_size = int(0.1 * size)
#    train_files = all_paths[:train_size]
#    val_files = all_paths[train_size:train_size + val_size]
#    test_files = all_paths[train_size+val_size:]
    
    # train_size = int(0.90 * size)
    # val_size = int(0.08 * size)
    train_files = all_paths[:]#train_size]
    val_files = val_paths[:] #train_size:train_size + val_size]
    #val_files = all_paths[train_size:]
    test_files = val_paths[:]#train_size+val_size:]
    print(f"train is {len(train_files)}, val is {len(val_files)}, test is {len(test_files)}")

    train_transform = transforms.Compose(
        [   
            #transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            #transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            #transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.CenterSpatialCropd(roi_size=[128, 192, 128],keys=["image", "label", "tumor"]),

            #transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[96, 96, 96],
            #                            random_size=False),
            #transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            #transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            #transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            #transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            #transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            
            #transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            #transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label","tumor"],),
        ]
    )
    val_transform = transforms.Compose(
        [   #transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            #transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.CenterSpatialCropd(roi_size=[128, 192, 128],keys=["image", "label", "tumor"]),

            #transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label", "tumor"]),
        ]
    )

    train_ds = PretrainDataset(train_files, transform=train_transform)

    val_ds = PretrainDataset(val_files, transform=val_transform)
    

    test_ds = PretrainDataset(test_files, transform=val_transform)

    loader = [train_ds, val_ds, test_ds]

    return loader

class get_loader_brats_folds:
    def __init__(self, data_dir, batch_size=1, folds=5, num_workers=2):
        
        all_dirs_train = os.listdir(data_dir+'trainning')
        all_dirs_validation = os.listdir(data_dir+'validation')
        
          
        
        all_paths = [os.path.join(data_dir+'trainning', d) for d in all_dirs_train]
        val_paths = [os.path.join(data_dir+'validation', d) for d in all_dirs_validation]

        #List all filenames (In the "training" folder as well as "validation" folder)
        self.all_paths = all_paths + val_paths 
    
           
        self.size = len(self.all_paths)
        

        np.random.seed(42)         #To make sure that the same shuffle is created everytime
        np.random.shuffle(self.all_paths)
        
        self.n_folds = folds
        self.fold_size = self.size//self.n_folds + 1
        
        #Create folds
        self.folds = [self.all_paths[i:i+self.fold_size] for i in range(0,self.size,self.fold_size)]
        
    
    def get_fold(self, fold_number): # Method only meant for Diffusion code
    
        # Get all the folds other than "fold_number" for training
        self.train_files = [x for x in self.all_paths if x not in self.folds[fold_number]]#train_size]
        print(f"train_mid: {self.train_files[42]}, train_end: {self.train_files[-1]}")

        # Get the fold corresponding to fold_number
        self.val_files = self.folds[fold_number] #train_size:train_size + val_size]
        print(f"val_beg: {self.val_files[0]}, val_end: {self.val_files[-1]}")
        #val_files = all_paths[train_size:]
        
        # Not relevant
        self.test_files = self.folds[fold_number]#train_size+val_size:]
        print(f"train is {len(self.train_files)}, val is {len(self.val_files)}, test is {len(self.test_files)}")
    
        # Define transforms
        train_transform = transforms.Compose(
            [   
                #transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
                #transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                #transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.CenterSpatialCropd(roi_size=[128, 192, 128],keys=["image", "label", "tumor"]),
    
                #transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[96, 96, 96],
                #                            random_size=False),
                #transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
                #transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                #transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                #transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                #transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                
                #transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                #transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                transforms.ToTensord(keys=["image", "label","tumor"],),
            ]
        )
        val_transform = transforms.Compose(
            [   #transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
                #transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.CenterSpatialCropd(roi_size=[128, 192, 128],keys=["image", "label", "tumor"]),
    
                #transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label", "tumor"]),
            ]
        )
    
        # Create Dataset object for Diffusion 
        train_ds = PretrainDataset(self.train_files, transform=train_transform)
    
        val_ds = PretrainDataset(self.val_files, transform=val_transform)
        
    
        test_ds = PretrainDataset(self.test_files, transform=val_transform)
    
        loader = [train_ds, val_ds, test_ds]
    
        return loader
    
    def get_fold_datapath(self, fold_number): #Method for 5-fold optimized UNet
       self.train_files = [x for x in self.all_paths if x not in self.folds[fold_number]]#train_size]
       print(f"train_mid: {self.train_files[42]}, train_end: {self.train_files[-1]}")
       self.val_files = self.folds[fold_number] #train_size:train_size + val_size]
       print(f"val_beg: {self.val_files[0]}, val_end: {self.val_files[-1]}")
       #val_files = all_paths[train_size:]
       self.test_files = self.folds[fold_number]#train_size+val_size:]
       print(f"train is {len(self.train_files)}, val is {len(self.val_files)}, test is {len(self.test_files)}")
   
 
   
       datapath = (self.train_files,self.val_files, self.test_files)
       
       
   
       return datapath
   
    def save_fold_info(self, fold_number):
      dict1 = {'train_files': self.train_files}
      dict2 = {'val_files': self.val_files}
      dict3 = {'test_files': self.test_files}
      df1, df2, df3 = pd.DataFrame(dict1), pd.DataFrame(dict2), pd.DataFrame(dict3)
      df1.to_csv(output_dir_withfold_number +'train_files.csv')
      df2.to_csv(output_dir_withfold_number +'val_files.csv')
      df3.to_csv(output_dir_withfold_number +'test_files.csv')
      
        