# Re-DiffiNet

This repository contains the implementation of our paper [(https://arxiv.org/abs/2402.07008)](https://openreview.net/pdf?id=vCEMidGVbv). The code is developed by Kurtlab for the 2023 International Brain Tumor Segmentations (BraTS) Cluster of Challenges (see [here](https://www.synapse.org/#!Synapse:syn51156910/wiki/621282).)

## Model architecure
The model architecure is shown below:
![Figure2](https://github.com/KurtLabUW/Re-DiffiNet/assets/9877397/68580b61-d30b-47ae-b236-139fea47a976)

## Data
We evaluated our method on the [BRATS2023 adult aglioma dataset]([https://www.med.upenn.edu/cbica/brats2020/data.html](https://www.synapse.org/#!Synapse:syn51156910/wiki/621282)).
The model input are picke files, and the each picke file coreespoinding to a subject with size of  **$`7*L*W*H `$**, where 7 represents concatenation of 4 MRI contrasts and  one-hot encolding label segmentation masks from U-Net model (t1, t2, t1ce, FLAIR, TC, WT, ET).

You can use your own U-Net model or use our previous Implementation of U-Net ([https://arxiv.org/abs/2402.07008](https://github.com/KurtLabUW/brats2023_updated)https://github.com/KurtLabUW/brats2023_updated). 

## Usage

1. Prepared the data accroding to the discription in the Data section.
2. To train the segmentation model, run the train.py. 
3. For model inference, run test.py 
