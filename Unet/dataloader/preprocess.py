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

root_dir='data/testdata'


def get_paths(label=False,axis='all',patches=False):
    folder='imgs'
    if label:
        folder='masks'
    paths=[]

    if patches:   
        print('Preparing patches')  
        path=os.path.join('data','testpatches','test1cortes',folder)
        paths.append(path)
    else:
        path1 =os.path.join(root_dir,'test1',folder)
        path2 =os.path.join(root_dir,'test2',folder)
        path3 =os.path.join(root_dir,'test3',folder)

        if axis==1:
            paths.append(path1)
        elif axis==2:
            paths.append(path2)
        elif axis==3:
            paths.append(path3)
        else:
            paths.append(path1);paths.append(path2);paths.append(path3)

    files=[]
    for path in paths:
        for file in sorted(list(Path(path).rglob('*.png'))):          
            # if label:
            #     if patches:
            #         destination = str(file).split('_')[0]+'.png'
            #         os.rename(file, destination)
            #     else:
            #         destination = str(file).split('_')[0]+'.png'
            #         os.rename(file, destination)

            files.append(file)

    print('lenfiles',len(files))
    
    return files

def create_datafile(patches=False,axis=1):
    patient=dict()
    axial_identifiers=zip(get_paths(axis=axis,patches=patches),get_paths(label=True,axis=axis,patches=patches))
    patient['axis']=[
                {"image": i, "label": j} for i,j in axial_identifiers]
    df_axis = pd.DataFrame(patient['axis'])
    if patches:
        df_axis.to_csv(os.path.join('data/testpatches','axis'+str(axis)+'patches.csv'),index=False)
    else:
        df_axis.to_csv(os.path.join(root_dir,'axis'+str(axis)+'.csv'), index=False)

create_datafile(patches=True,axis=1)