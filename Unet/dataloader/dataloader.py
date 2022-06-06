import os
import time
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
import glob
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class FeTa_data(Dataset):
    def __init__(self,csv_file,root_dir,patches=False):

        super(FeTa_data, self).__init__()

        self.csv_file = csv_file
        self.root_dir = root_dir
        self.patches = patches
        self.filenames = pd.read_csv(os.path.join(root_dir,csv_file))
        self.filenames['patient']=self.filenames['image'].apply(lambda x: x.split('/')[-2])
        self.filenames['slice']=self.filenames['image'].apply(lambda x: x.split('/')[-1].split('-')[-1].split('.')[0])
        self.filenames['slice']=self.filenames['slice'].astype(int)
        self.filenames=self.filenames.sort_values(by=['patient','slice'])
        self.patients = self.filenames['patient'].unique()

        if self.patches:
            self.filenames['patch']=self.filenames['image'].apply(lambda x: x.split('/')[-1].split('-')[-1].split('.')[0])
            self.filenames['slice']=self.filenames['image'].apply(lambda x: x.split('/')[-1].split('-')[-2])
            self.filenames['slice']=self.filenames['slice'].astype(int)
            self.filenames['patch']=self.filenames['patch'].astype(int)
      
            self.filenames=self.filenames.sort_values(by=['patient','slice','patch'])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, patient ,slice,patch=None):
    
        def preprocess(pil_img, scale, is_mask):
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
            img_ndarray = np.asarray(pil_img)

            if not is_mask:
                if img_ndarray.ndim == 2:
                    img_ndarray = img_ndarray[np.newaxis, ...]
                else:
                    img_ndarray = img_ndarray.transpose((2, 0, 1))

                img_ndarray = img_ndarray / 255

            return img_ndarray

        #filter filenames by patient
        self.names = self.filenames[self.filenames['patient'] == patient]

        imagepath , labelpath = self.names[self.names['slice']==slice]['image'].values[0], self.names[self.names['slice']==slice]['label'].values[0]


        if self.patches:
            self.names = self.names[self.names['slice']==slice]
            imagepath , labelpath = self.names[self.names['patch']==patch]['image'].values[0], self.names[self.names['patch']==patch]['label'].values[0]

        name = imagepath.split('/')[-1]
        image,label= Image.open(imagepath),  Image.open(labelpath)
        image,label= preprocess(image, 1, 0),preprocess(label, 1, 1)
        return image,label,name
        
    
    def get_patients(self):
        return self.patients

# #root_dir='data/testdata'
# root_dir='data/testpatches'

# ds= FeTa_data('axis1patches.csv',root_dir,patches=True)
# image,label,name = ds.__getitem__('23',0,5)
# print(image.shape)
# print(label.shape)
# print(name)