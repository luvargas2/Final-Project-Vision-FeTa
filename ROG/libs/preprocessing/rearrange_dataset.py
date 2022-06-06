import os
import glob
import numpy as np
import pandas as pd
import pickle
import shutil
from batchgenerators.utilities.file_and_folder_operations import *

data_root='/home/lfvargas10/ProyectoVision/feta_2.1'
with open('/home/lfvargas10/ProyectoVision/Train.dat', 'rb') as f:
    train=pickle.load(f)
with open('/home/lfvargas10/ProyectoVision/Val.dat', 'rb') as f:
    val=pickle.load(f)
with open('/home/lfvargas10/ProyectoVision/Test.dat', 'rb') as f:
    imagesTs=pickle.load(f)

imagesTr= train + val

#Create new dataset folders

#si no existe el directorio lo crea
newTrain_root=os.path.join('/home/lfvargas10/ProyectoVision', 'Data','Task15_feta', 'imagesTr')
newTest_root=os.path.join('/home/lfvargas10/ProyectoVision','Data', 'Task15_feta', 'imagesTs')
newMask_root=os.path.join('/home/lfvargas10/ProyectoVision','Data', 'Task15_feta', 'labelsTr')
predMask_root=os.path.join('/home/lfvargas10/ProyectoVision','Data', 'Task15_feta', 'labelsTs')


def create_folder(split,root):
    if not os.path.exists(root):
        os.mkdir(root)

    if split=='test':
        if not os.path.exists(predMask_root):
            os.mkdir(predMask_root)
        mask_root= predMask_root
        split_images=imagesTs
    else:
        if not os.path.exists(newMask_root):
            os.mkdir(newMask_root)
        mask_root= newMask_root
        split_images=imagesTr


    for i in split_images:
        if i < 10:
            imagen = glob.glob(os.path.join(data_root, f'sub-00{i}', 'anat', '*T2w.nii.gz'))[0]
            mask = glob.glob(os.path.join(data_root, f'sub-00{i}', 'anat', '*dseg.nii.gz'))[0]
            shutil.copyfile(imagen, os.path.join(root, f'sub-00{i}_rec-mial_T2w.nii.gz'))
            shutil.copyfile(mask, os.path.join(mask_root, f'sub-00{i}_rec-mial_T2w.nii.gz'))
        else:
            imagen = glob.glob(os.path.join(data_root, f'sub-0{i}', 'anat', '*T2w.nii.gz'))[0]
            mask = glob.glob(os.path.join(data_root, f'sub-0{i}', 'anat', '*dseg.nii.gz'))[0]
            shutil.copyfile(imagen, os.path.join(root, f'sub-0{i}_rec-mial_T2w.nii.gz'))
            shutil.copyfile(mask, os.path.join(mask_root, f'sub-0{i}_rec-mial_T2w.nii.gz'))

def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-7] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques

def help_datasetjson():

    train_identifiers = get_identifiers_from_splitted_files(newTrain_root)
    test_identifiers = get_identifiers_from_splitted_files(newTest_root)

    json_dict = dict()
    json_dict['training'] = [
            {"image": "imagesTr/%s.nii.gz" % i, "label": "labelsTr/%s.nii.gz" % i} for i
            in
            train_identifiers]

    #Create a dataframe of the json_dict and save it as a csv file
    
    df_train = pd.DataFrame(json_dict['training'])
    #df_train.to_csv('/home/lfvargas10/ProyectoVision/ROG/Tasks/Task15_feta/train_fold0.csv', index=False)

    json_dict['test'] = [ {"image": "imagesTs/%s.nii.gz" % i, "label": "labelsTs/%s.nii.gz" % i} for i in test_identifiers]
    print(json_dict['test'])
    df_test = pd.DataFrame(json_dict['test'])
    #df_test.to_csv('/home/lfvargas10/ProyectoVision/ROG/Tasks/Task15_feta/test_fold0.csv', index=False)

    return json_dict

#create_folder('train', newTrain_root)

#create_folder('test', newTest_root)

help_datasetjson()

