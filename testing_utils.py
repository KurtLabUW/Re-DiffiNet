# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:52:59 2023

@author: Kurtlab
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

def create_custom_colormap():
    # Define the colors for your colormap
    colors = [(1.0, 0.5, 0.0),  # Orange for 1.0
              (1.0, 1.0, 1.0),  # White for 0                                
              (1.0, 0.5, 0.5),  # Light red for -0.5
              (0.0, 0.5, 1.0),  # Light blue for -0.1
              (0.0, 0.0, 1.0),  # Blue for 0.1
              (1.0, 0.0, 0.0),  # Red for  0.5
              (0.0, 0.0, 0.0)]  # Black for -1.0
    
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
    
    return cmap



def plotfigures(image,t0,t1,t2,o0,o1,o2,title):
    nslice = 75;
    # vislize the output
    # if len(np.bincount(np.nonzero(o2[  0, :, :, :])[2])) != 0:
    #     nslice = np.argmax(np.bincount(np.nonzero(o2[  :, :, :, :])[2]))
    # else:
    #     nslice = 100   
    
    diff = np.absolute(t0[0,  :, :, nslice]-o0[0,  :, :, nslice])
    
    
    
    plt.figure(figsize=(15, 9), dpi=1000)

    plt.suptitle(title)
    plt.subplot(4, 4, 1)
    plt.imshow(image[ 0,0, :, :, nslice], cmap='gray')
    #plt.colorbar() 
    plt.title('Modality 1')
    plt.grid(False)
    plt.axis('off')
 
    plt.subplot(4, 4, 2)
    plt.imshow(image[ 0,1, :, :, nslice], cmap='gray')
    #plt.colorbar()    
    plt.title('Modality 2')
    plt.grid(False)
    plt.axis('off')

    plt.subplot(4, 4, 3)
    plt.imshow(image[ 0,2, :, :, nslice], cmap='gray')
    #plt.colorbar()
    plt.title('Modality 3') 
    plt.grid(False)
    plt.axis('off')
    plt.subplot(4, 4, 4)
    plt.imshow(image[ 0,3, :, :, nslice], cmap='gray')
    #plt.colorbar()
    plt.title('Modality 4') 
    plt.grid(False)
    plt.axis('off')
    
    
    plt.subplot(4, 4, 5)
    plt.imshow(o0[ 0, :, :, nslice])
    #plt.colorbar()
    plt.title('Diff Prediction label 1')
    plt.grid(False)
    plt.axis('off')

    plt.subplot(4, 4, 6)
    plt.imshow(o1[ 0, :, :, nslice])
    #plt.colorbar()
    plt.title('Diff rediction label 2')
    plt.grid(False)
    plt.axis('off')

    plt.subplot(4, 4, 7)
    plt.imshow(o2[ 0, :, :, nslice])
    #plt.colorbar()
    plt.title('Diff Prediction label 3')
    plt.grid(False)
    plt.axis('off')
    o_3channel =np.stack((o0, o1, o2),axis=-1)
    o_3channel[:,:,:,:,0] = o_3channel[:,:,:,:,0] -o_3channel[:,:,:,:,1]
    o_3channel[:,:,:,:,1] = o_3channel[:,:,:,:,1] -o_3channel[:,:,:,:,2]
    o_3channel = np.stack((o_3channel[:,:,:,:,1], o_3channel[:,:,:,:,0], o_3channel[:,:,:,:,2]),axis=-1)
    
    plt.subplot(3, 4, 8)
    plt.imshow(o_3channel[ 0, :, :, nslice,:])
    #plt.colorbar()
    plt.title('overlay')
    plt.grid(False)
    plt.axis('off')
    
    
    plt.subplot(4, 4, 9)
    plt.imshow(t0[0,  :, :, nslice])
    #plt.colorbar()
    plt.title('Ground Truth 1')
    plt.grid(False)
    plt.axis('off')

    plt.subplot(4, 4, 10)
    plt.imshow(t1[0,  :, :, nslice])
    #plt.colorbar()
    plt.title('Ground Truth 2')
    plt.grid(False)
    plt.axis('off')

    plt.subplot(4, 4, 11)
    plt.imshow(t2[0,  :, :, nslice])
    plt.title('Ground Truth 3')
    #plt.colorbar()
    plt.grid(False)
    plt.axis('off')
    
    t_3channel =np.stack((t0, t1, t2),axis=-1)
    t_3channel[:,:,:,:,0] = t_3channel[:,:,:,:,0] - t_3channel[:,:,:,:,1] 
    t_3channel[:,:,:,:,1] = t_3channel[:,:,:,:,1] - t_3channel[:,:,:,:,2]
    t_3channel = np.stack((t_3channel[:,:,:,:,1], t_3channel[:,:,:,:,0], t_3channel[:,:,:,:,2]),axis=-1)
    
    
    plt.subplot(3, 4, 12)
    plt.imshow(t_3channel[0,  :, :, nslice])
    #plt.colorbar()
    plt.title('overlay') 
    plt.grid(False)
    plt.axis('off')
    
    
    # plt.subplot(4, 4, 13)
    # plt.imshow(np.absolute(t0[0,  :, :, nslice]-o0[0,  :, :, nslice]))
    # plt.colorbar()
    # plt.title('Difference 1')

    # plt.subplot(4, 4, 14)
    # plt.imshow(np.absolute(t1[0,  :, :, nslice]-o1[0,  :, :, nslice]))
    # plt.colorbar()
    # plt.title('Difference 2')

    # plt.subplot(4, 4, 15)
    # plt.imshow(np.absolute(t2[0,  :, :, nslice]-o2[0,  :, :, nslice]))
    # plt.colorbar()
    # plt.title('Difference 3') 

    plt.show()   
    


# def plotfigures(image,t0,t1,t2,o0,o1,o2,title):
#     nslice = 75;
#     # vislize the output
#     # if len(np.bincount(np.nonzero(o2[  0, :, :, :])[2])) != 0:
#     #     nslice = np.argmax(np.bincount(np.nonzero(o2[  :, :, :, :])[2]))
#     # else:
#     #     nslice = 100   
    
#     diff = np.absolute(t0[0,  :, :, nslice]-o0[0,  :, :, nslice])
    
    
    
#     plt.figure(figsize=(12, 9), dpi=1000)
#     plt.suptitle(title)
#     plt.subplot(4, 4, 1)
#     plt.imshow(image[ 0,0, :, :, nslice], cmap='gray')
#     plt.colorbar() 
#     plt.title('Modality 1')
 
#     plt.subplot(4, 4, 2)
#     plt.imshow(image[ 0,1, :, :, nslice], cmap='gray')
#     plt.colorbar()    
#     plt.title('Modality 2')

#     plt.subplot(4, 4, 3)
#     plt.imshow(image[ 0,2, :, :, nslice], cmap='gray')
#     plt.colorbar()
#     plt.title('Modality 3')  
#     # plt.subplot(3, 4, 4)
#     # plt.imshow(image[ 0,3, :, :, nslice], cmap='gray')
#     # plt.title('Prediction label 3')  
    
    
#     plt.subplot(4, 4, 5)
#     plt.imshow(o0[ 0, :, :, nslice])
#     plt.colorbar()
#     plt.title('Diff Prediction label 1')

#     plt.subplot(4, 4, 6)
#     plt.imshow(o1[ 0, :, :, nslice])
#     plt.colorbar()
#     plt.title('Diff rediction label 2')

#     plt.subplot(4, 4, 7)
#     plt.imshow(o2[ 0, :, :, nslice])
#     plt.colorbar()
#     plt.title('Diff Prediction label 3')
#     # o_3channel =np.stack((o0, o1, o2),axis=-1)
#     # o_3channel[:,:,:,:,1] = o_3channel[:,:,:,:,1] -o_3channel[:,:,:,:,0]
#     # o_3channel[:,:,:,:,0] = o_3channel[:,:,:,:,0] -o_3channel[:,:,:,:,2]
    
#     # plt.subplot(3, 4, 8)
#     # plt.imshow(o_3channel[ 0, nslice, :, :])
#     # plt.title('Prediction label 3')
    
    
#     plt.subplot(4, 4, 9)
#     plt.imshow(t0[0,  :, :, nslice])
#     plt.colorbar()
#     plt.title('Ground Truth 1')

#     plt.subplot(4, 4, 10)
#     plt.imshow(t1[0,  :, :, nslice])
#     plt.colorbar()
#     plt.title('Ground Truth 2')

#     plt.subplot(4, 4, 11)
#     plt.imshow(t2[0,  :, :, nslice])
#     plt.title('Ground Truth 3')
#     plt.colorbar()
#     # t_3channel =np.stack((t0, t1, t2),axis=-1)
#     # t_3channel[:,:,:,:,1] = t_3channel[:,:,:,:,1] - t_3channel[:,:,:,:,0]
#     # t_3channel[:,:,:,:,0] = t_3channel[:,:,:,:,0] - t_3channel[:,:,:,:,2] 
    
#     # plt.subplot(3, 4, 12)
#     # plt.imshow(t_3channel[0,  nslice, :, :])
#     # plt.title('Ground Trueth 3') 
    
    
#     plt.subplot(4, 4, 13)
#     plt.imshow(np.absolute(t0[0,  :, :, nslice]-o0[0,  :, :, nslice]))
#     plt.colorbar()
#     plt.title('Difference 1')

#     plt.subplot(4, 4, 14)
#     plt.imshow(np.absolute(t1[0,  :, :, nslice]-o1[0,  :, :, nslice]))
#     plt.colorbar()
#     plt.title('Difference 2')

#     plt.subplot(4, 4, 15)
#     plt.imshow(np.absolute(t2[0,  :, :, nslice]-o2[0,  :, :, nslice]))
#     plt.colorbar()
#     plt.title('Difference 3') 

#     plt.show()   
    
    
    
def plotfigures_changes(filename, nslice, image,t0,t1,t2,o0,o1,o2,ot0,ot1,ot2,dice_diffwt,dice_difftc,dice_diffet):
    


        # vislize the output
        # if len(np.bincount(np.nonzero(o2[  0, :, :, :])[2])) != 0:
        #     nslice = np.argmax(np.bincount(np.nonzero(o2[  :, :, :, :])[2]))
        # else:
        #     nslice = 100   
        
        # diff = np.absolute(t0[0,  :, :, nslice]-o0[0,  :, :, nslice])
        
        # Create the custom colormap
        #new_inferno = create_custom_colormap()
        
        new_inferno = ListedColormap(['orange', 'navy', 'lime','white','green','blue','coral'], 'indexed')
        
        new_inferno1 = ListedColormap(['navy','white','mediumslateblue'], 'indexed')
        
        
        plt.figure(figsize=(12, 9), dpi=300)
        plt.subplot(4, 4, 1)
        plt.imshow(image[ 0,0, :, :, nslice], cmap='gray')
        plt.colorbar()
        plt.title('Ground Truth 1')
        plt.subplot(4, 4, 2)
        plt.imshow(image[ 0,1, :, :, nslice], cmap='gray')
        plt.colorbar()
        plt.title('Ground Truth 2')
        plt.subplot(4, 4, 3)
        plt.imshow(image[ 0,2, :, :, nslice], cmap='gray')
        plt.colorbar()
        plt.title('Ground Truth 3')  
        # plt.subplot(3, 4, 4)
        # plt.imshow(image[ 0,3, :, :, nslice], cmap='gray')
        # plt.title('Prediction label 3')  

        #new_inferno = cm.get_cmap('inferno', 7)
        
        plt.subplot(4, 4, 5)
        plt.imshow(o0[  :, :, nslice], cmap=new_inferno1, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Delta Diff label 1')
        plt.subplot(4, 4, 6)
        plt.imshow(o1[ :, :, nslice], cmap=new_inferno1, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Delta Diff label 2')
        plt.subplot(4, 4, 7)
        plt.imshow(o2[ :, :, nslice], cmap=new_inferno1, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Delta Diff label 3')
        # o_3channel =np.stack((o0, o1, o2),axis=-1)
        # o_3channel[:,:,:,:,1] = o_3channel[:,:,:,:,1] -o_3channel[:,:,:,:,0]
        # o_3channel[:,:,:,:,0] = o_3channel[:,:,:,:,0] -o_3channel[:,:,:,:,2]
        
        # plt.subplot(3, 4, 8)
        # plt.imshow(o_3channel[ 0, nslice, :, :])
        # plt.title('Prediction label 3')
        
        
        plt.subplot(4, 4, 9)
        plt.imshow(t0[  :, :, nslice], cmap=new_inferno1, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Delta UNet 1')
        plt.subplot(4, 4, 10)
        plt.imshow(t1[  :, :, nslice], cmap=new_inferno1, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Delta UNet 2')
        plt.subplot(4, 4, 11)
        plt.imshow(t2[  :, :, nslice], cmap=new_inferno1, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Delta UNet 3')
        # t_3channel =np.stack((t0, t1, t2),axis=-1)
        # t_3channel[:,:,:,:,1] = t_3channel[:,:,:,:,1] - t_3channel[:,:,:,:,0]
        # t_3channel[:,:,:,:,0] = t_3channel[:,:,:,:,0] - t_3channel[:,:,:,:,2] 
        
        # plt.subplot(3, 4, 12)
        # plt.imshow(t_3channel[0,  nslice, :, :])
        # plt.title('Ground Trueth 3') 
        
        
        plt.subplot(4, 4, 13)
        plt.imshow(ot0[  :, :, nslice], cmap=new_inferno, vmin=1, vmax=7)
        plt.colorbar()
        plt.title('Change Mat TC:'+ str(round(dice_difftc,2)))
        plt.subplot(4, 4, 14)
        plt.imshow(ot1[  :, :, nslice], cmap=new_inferno, vmin=1, vmax=7)
        plt.colorbar()
        plt.title('Change Mat WT:'+ str(round(dice_diffwt,2)))
        plt.subplot(4, 4, 15)
        plt.imshow(ot2[  :, :, nslice], cmap=new_inferno, vmin=1, vmax=7)
        plt.colorbar()
        plt.title('Change Mat ET:'+ str(round(dice_diffet,2))) 
        plt.show()       
        
        