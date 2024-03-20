# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:55:24 2023

@author: Kurtlab
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:46:21 2023

@author: Kurtlab
"""

import numpy as np
from brats_data_utils_multi_label_5folds import get_loader_brats_folds
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice, hausdorff_distance_95, recall, fscore
import argparse
import yaml 
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import matplotlib.pyplot as plt
set_determinism(123)
import os
from testing_utils import plotfigures, plotfigures_changes
import nibabel as nib
from scipy import ndimage as ndi
from monai import transforms 
import scipy
import pandas as pd



data_dir = 'Z:/Brats2023/diff_cv_data/fold1/'
output_dir ='Z:/Brats2023/diff_cv_results/delta/fold1_test2/'
print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")

os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:1"

max_epoch = 300
batch_size = 1
val_every = 10
device = "cuda:1"

dict1 = {'filename': [], 'wt':[],
                'tc':[],
                'et':[],
                'v_total':[],
                'v_nonzeros':[],
                'x_wt':[],
                'y_wt':[],
                'z_wt':[],
                'x_et':[],
                'y_et':[],
                'z_et':[],
                'x_tc':[],
                'y_tc':[],
                'z_tc':[],
                'diff-unetwt':[],
                'diff-unettc':[],
                'diff-unetet':[],
                'diffwt':[],
                'difftc':[],
                'diffet':[],
                'diffwt_hd':[],
                'difftc_hd':[],
                'diffet_hd':[],
                'unetwt':[],
                'unettc':[],
                'unetet':[],
                'unetwt_hd':[],
                'unettc_hd':[],
                'unetet_hd':[],
               }
          
df = pd.DataFrame(dict1)


def flood_fill_hull(batch):
    images = []
    for i in range(batch.shape[0]):
        image = batch[i,i,:,:,:] # N x C x D x W x H    
        points = np.transpose(np.where(image))
        if points.size > 4:
            try:
                hull = scipy.spatial.ConvexHull(points)
                deln = scipy.spatial.Delaunay(points[hull.vertices]) 
                idx = np.stack(np.indices(image.shape), axis = -1)
                out_idx = np.nonzero(deln.find_simplex(idx) + 1)
                out_img = np.zeros(image.shape)
                out_img[out_idx] = 1
                images.append(out_img)
                out_img = np.stack(images,axis=0)
            except:
                out_img = image
        else:
            out_img = image
    

    return out_img

cropper = transforms.CropForeground()
def convex_hull_cropper(image):    
    #cropping based on largest covex hull for a tumor in the batch
    
    image = flood_fill_hull(image) # calculate convex hull
    image = cropper(image)
    
    return image

use_UNetopt = True

if use_UNetopt:
    number_modality = 7
else:
    number_modality = 4
    
number_targets = 3 ## WT, TC, ET

def is_outlier(points, thresh=6):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def reshape_input(input):
    out = np.zeros((240, 240, 155))
    out[56:184,24:216,14:142] = input 
    return out

def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embedding=embedding)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            uncer_step = 10 
            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 128, 192, 128), model_kwargs={"image": image, "embeddings": embeddings}))

            sample_return = torch.zeros((1, number_targets, 128, 192, 128))
            correction_map = torch.zeros((1, number_targets, 128, 192, 128))

            for index in range(9,10):
            #for index in range(7,10):
# 
                uncer_out = 0
                for i in range(uncer_step):
                    uncer_out += sample_outputs[i]["all_model_outputs"][index]
                    # uncer_out += torch.sigmoid(sample_outputs[i]["all_model_outputs"][index]) #UNCER USING PROBS
                uncer_out = uncer_out / uncer_step
                uncer = compute_uncer(uncer_out).cpu()
                
                
                
                
                
                # if uncer[voxel] > value : correction_map[voxel] -> 0 else correction_map[voxel] -> 1
                
                # if uncer > threshold, ignore this diffusion step
                correction_map = torch.where(uncer >0.004, 0, 1) # uncertainty values
                
                # if index < threshold, ignore this diffusion step
                w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / 10)) * (1 - uncer))
                #w = torch.exp(torch.sigmoid(torch.tensor((np.where(index < 6, 0, index) + 1) / 10)) * (1 - uncer))
              
                for i in range(uncer_step):
                    sample_return += w * sample_outputs[i]["all_samples"][index].cpu()
                    #sample_return += correction_map*w*sample_outputs[i]["all_samples"][index].cpu()
                    # sample_return1 = sample_return
                   
            # plotfigures(image.cpu(),uncer[:,0,:,:,:],uncer[:,1,:,:,:],uncer[:,2,:,:,:],sample_return[:,0,:,:,:],sample_return[:,1,:,:,:],sample_return[:,2,:,:,:])

            return sample_return

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[128, 192, 128],
                                        sw_batch_size=1,
                                        overlap=0.5)
        
        self.model = DiffUNet()
        self.test_dir = test_dir
        

                  

    def get_input(self, batch):
        
            
        label = batch["label"]
        image = batch["image"]  
        tumor = batch["tumor"]
       
        label = label.float()
        return image, label, tumor.float() 

    def validation_step(self, batch):
        image, label, tumor = self.get_input(batch)
        #print(label)
        ########## output delta
        delta_output = self.window_infer(image, self.model, pred_type="ddim_sample")
        delta_output = torch.sigmoid(delta_output)
        image = image.float().cpu().numpy()
        delta_output = (delta_output > 0.5).float().cpu().numpy()
        
        ########## change the unet prediction based on output delta
        unet_output = image[0,4:7,:,:,:]
        output = np.zeros_like(label) #
        output = np.where(delta_output==0,unet_output,-1)
        output = np.where(delta_output!=0,np.abs(unet_output-delta_output),output)
        
        ######### visulize the change mat        
        delta_unet= unet_output - tumor[0,:,:,:,:]
        delta_unet = np.where(delta_unet==0,0.5,delta_unet)
        delta_diff=output[0,:,:,:,:]  - tumor[0,:,:,:,:]
        delta_diff = np.where(delta_diff==0,0.1,delta_diff)
        
        change_mat = delta_diff*delta_unet
        change_mat = np.where(change_mat==1.,7,change_mat)
        change_mat = np.where(change_mat==0.5,6,change_mat)
        change_mat = np.where(change_mat==0.1,5,change_mat)
        change_mat = np.where(change_mat==0.05,4,change_mat)
        change_mat = np.where(change_mat==-0.1,3,change_mat)
        change_mat = np.where(change_mat==-0.5,2,change_mat)
        change_mat = np.where(change_mat==-1.,1,change_mat)
        
        delta_unet = np.where(delta_unet==0.5,0,delta_unet)
        delta_diff = np.where(delta_diff==0.1,0,delta_diff)              
        # change_mat = np.where(change_mat==0.05,1,change_mat)
        # change_mat = np.where(change_mat==0.1,0,change_mat)
        # change_mat = np.where(change_mat==-0.1,0,change_mat)
        print('output_unique',np.unique(output))
        print('label_unique',np.unique(label))
        
    
        ########## calculate dice and hd score
        # whole tumor
        target = tumor.float().cpu().numpy()
        o1 = output[:, 1]
        t1 = target[:, 1] # ce
        wt = dice(o1, t1)
        wt_hd = hausdorff_distance_95(o1, t1)
        wt_recall = recall(o1, t1)

        # tumor core
        o0 = output[:, 0]
        t0 = target[:, 0]
        tc = dice(o0, t0)
        tc_hd = hausdorff_distance_95(o0, t0)
        tc_recall = recall(o0, t0)

        # enhancing tumor 
        o2 = output[:, 2]
        t2 = target[:, 2]
       
        et = dice(o2, t2)
        et_hd = hausdorff_distance_95(o2, t2)
        et_recall = recall(o2, t2)
        
        
        
        dice_diffwt = wt-dice(np.expand_dims(unet_output[1,], axis=0), t1)
        dice_difftc = tc-dice(np.expand_dims(unet_output[0,], axis=0), t0)
        dice_diffet = et-dice(np.expand_dims(unet_output[2,], axis=0), t2)
        
        

        ########## calculate dice and hd score of 
        # whole tumor
        target = tumor.float().cpu().numpy()
        o11 = unet_output[[1]]
        unetwt = dice(o11, t1)
        unetwt_hd = hausdorff_distance_95(o11, t1)


        # tumor core
        o00 = unet_output[[0]]
        unettc = dice(o00, t0)
        unettc_hd = hausdorff_distance_95(o00, t0)
        

        # enhancing tumor 
        o22 = unet_output[[2]]
        unetet = dice(o22, t2)
        unetet_hd = hausdorff_distance_95(o22, t2)
             
      
        
        # output figures and data
    
        plotfigures(image,t0,t1,t2,o0,o1,o2,'diffusion')
        
        plotfigures(image,t0,t1,t2,o00,o11,o22,'unet')
        
        #plotfigures(image,np.abs(t0-o00),np.abs(t1-o11),np.abs(t2-o22),np.abs(o0-o00),np.abs(o1-o11),np.abs(o2-o22),'discrepancy')
        
        
        slices = [45, 60, 75, 90]

        

        print(f"wt is {wt}, tc is {tc}, et is {et}")
        
    

        # save file for validation
        brats_loader = get_loader_brats_folds(data_dir=data_dir, batch_size=batch_size, folds= 5)
        _, val_dir, _ = brats_loader.get_fold_datapath(0)
        val_dir = [s.split('\\')[1] for s in val_dir]
        #input_dir = data_dir+'/validation'
        filenames = val_dir
        num = len(os.listdir(output_dir))

        
        filename_output = filenames[num].split(".")[0] + '.nii.gz'
        filename_images = filenames[num].split(".")[0] + '.png'
        
        # for nslice in slices:
        #     plotfigures_changes(filename_images, nslice, target,
        #                         delta_unet[0,:],delta_unet[1,:],delta_unet[2,:],
        #                         delta_diff[0,:],delta_diff[1,:],delta_diff[2,:], 
        #                         change_mat[0,:], change_mat[1,:], change_mat[2,:],
        #                         dice_diffwt,dice_difftc,dice_diffet)
                
        
        
        output = np.squeeze(output).astype(np.uint8)
        output_for_nifti = np.zeros_like(output[0])
        output[2]=output[2]
        output[1]=output[1]- output[0]
        output[0]=output[0]- output[2]
        
        wholetumor = output[0] + output[1] + output[2]
        wholetumor = np.expand_dims(wholetumor, axis=(0))
        wholetumor = np.expand_dims(wholetumor, axis=(0))
        wholetumor = convex_hull_cropper(wholetumor)
        
        enhancingtumor = output[2]
        enhancingtumor = np.expand_dims(enhancingtumor, axis=(0))
        enhancingtumor = np.expand_dims(enhancingtumor, axis=(0))
        enhancingtumor = convex_hull_cropper(enhancingtumor)
        
        tumorcore = output[0] + output[2]
        tumorcore = np.expand_dims(tumorcore, axis=(0))
        tumorcore = np.expand_dims(tumorcore, axis=(0))
        tumorcore = convex_hull_cropper(tumorcore)
         
        
        x_wt = wholetumor.shape[0]
        y_wt = wholetumor.shape[1]
        z_wt = wholetumor.shape[2]
        
        x_et = enhancingtumor.shape[0]
        y_et = enhancingtumor.shape[1]
        z_et = enhancingtumor.shape[2]
        
        x_tc = tumorcore.shape[0]
        y_tc = tumorcore.shape[1]
        z_tc = tumorcore.shape[2]

        v_total = np.size(wholetumor)        
        v_nonzeros = np.count_nonzero(wholetumor)
        print(f"v_total is {v_total}, v_nonzeros is {v_nonzeros}")
        
        output_for_nifti = output[0] + output[1]*2 + output[2]*3
        output_for_nifti = np.squeeze(output_for_nifti)
        reshaped_nifti = reshape_input(output_for_nifti)
        nifti_img = nib.nifti1.Nifti1Image(reshaped_nifti, affine=None)
        
        nib.nifti1.save(nifti_img, os.path.join(output_dir, filename_output)) 
        
                  
        df.loc[len(df.index)] = [filenames[num].split(".")[0], wt, tc, et, v_total, v_nonzeros,
                                 x_wt,y_wt,z_wt,x_et,y_et,z_et,x_tc,y_tc,z_tc,
                                 dice_diffwt,dice_difftc,dice_diffet,wt,tc,et,wt_hd, tc_hd, et_hd,unetwt,unettc,unetet,unetwt_hd,unettc_hd,unetet_hd]
        
        
        

        
                 
        return [wt, tc, et, wt_hd, tc_hd, et_hd,unetwt,unettc,unetet,unetwt_hd,unettc_hd,unetet_hd]
      
        
        
        

if __name__ == "__main__":

    n_folds = 5  
    fold_number=1
    brats_loader = get_loader_brats_folds(data_dir=data_dir, batch_size=batch_size, folds= n_folds)
    #for fold_number in range(n_folds):
    train_ds, val_ds, test_ds = brats_loader.get_fold(fold_number)
    in_dir, val_dir, test_dir = brats_loader.get_fold_datapath(fold_number)
    trainer = BraTSTrainer(env_type="pytorch",
                                    max_epochs=max_epoch,
                                    batch_size=batch_size,
                                    device=device,
                                    val_every=val_every,
                                    num_gpus=1,
                                    master_port=17751,
                                    training_script=__file__)


    logdir = 'C:/Users/Kurtlab/Documents/TianyiProject/logs/delta/fold1/model/best_model_0.3707.pt'
    trainer.load_state_dict(logdir)
    v_mean, _ = trainer.validation_single_gpu(val_dataset=test_ds)
    
    df.to_csv(output_dir +'out_fold1.csv')

    print(f"v_mean is {v_mean}")
