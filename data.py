import numpy as np
from PIL import Image
import glob
import torch
from torch.utils.data.dataset import Dataset
import os

'''
Pytorch uses datasets and has a very handy way of creatig dataloaders in your main.py
Make sure you read enough documentation.
'''

class Data(Dataset):
    def __init__(self, data_dir):
        #gets the data from the directory
        self.image_list = glob.glob(data_dir+'/*')

        #calculates the length of image_list
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index]
        # Open image
        image = Image.open(single_image_path)

        # Do some operations on image
        # Convert to numpy, dim = 28x28
        image_np = np.asarray(image)/255
        #image_np = np.rot90(image_np).T

        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension

#         image_np = np.expand_dims(image_np, 0)
        '''
        #TODO: Convert your numpy to a tensor and get the labels
        '''
    
    
        image_tensor = torch.from_numpy(image_np.copy()).float()

        #try janky rotate
        image_tensor = torch.rot90(image_tensor, 3).T

        class_indicator_location = single_image_path.rfind('_c')
        rot_indicator_location = single_image_path.rfind('_r')
        class_label = int(single_image_path[class_indicator_location+2:class_indicator_location+3])
        rot_label= int(single_image_path[rot_indicator_location+2:rot_indicator_location+3])
        return (image_tensor, class_label, rot_label)

    def __len__(self):
        return self.data_len
