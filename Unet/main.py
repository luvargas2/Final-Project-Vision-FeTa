#Import general libraries
from pathlib import Path
import argparse
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import pickle
import glob
import cv2
from sklearn.metrics import jaccard_score

#import pytorch libraries
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# import utils 
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from unet import UNet
from dataloader.dataloader import FeTa_data
import test
import testUnet2D


def main(args):
    print(f'-----Running FeTA Multi-tissue segmentation model on device {args.gpu}-----')
    experiment = args.experiment

    # PATHS AND DIRS
    args.save_path = os.path.join('Results',args.nameUnet)
    os.makedirs(args.save_path, exist_ok=True)

    if experiment == 'unet2D':
        # CREATE THE NETWORK ARCHITECTURE
        net1 = UNet(n_channels=1, n_classes=8, bilinear=False)
        device = torch.device('cuda:%d' % int(args.gpu))
        net1.load_state_dict(torch.load(os.path.join('models', 'unet2Dalbcortes24.pth'),map_location=device))
        net1.to(device=device)

        ##TEST/DEMO

        if args.mode=='demo':
            testUnet2D.test(int(args.gpu),demo=True)
        else:
            testUnet2D.test(int(args.gpu))
  
       
    elif experiment == 'unet25D':
        # CREATE THE NETWORK ARCHITECTURE
        net1 = UNet(n_channels=1, n_classes=8, bilinear=False)
        device = torch.device('cuda:%d' % int(args.gpu))
        net1.load_state_dict(torch.load(os.path.join('models', 'unet25Dcombe50.pth'),map_location=device))
        net1.to(device=device)
        
        net2 = UNet(n_channels=1, n_classes=8, bilinear=False)
        device = torch.device('cuda:%d' % int(args.gpu))
        net2.load_state_dict(torch.load(os.path.join('models','unet25Dcombe50.pth'),map_location=device))
        net2.to(device=device)

        net3 = UNet(n_channels=1, n_classes=8, bilinear=False)
        device = torch.device('cuda:%d' % int(args.gpu))
        net3.load_state_dict(torch.load(os.path.join('models', 'unet25Dcombe50.pth'),map_location=device))
        net3.to(device=device)

         # DATASETS
        test_dataset_1 = FeTa_data('axis1.csv',root_dir='data/testdata')
        test_dataset_2 = FeTa_data('axis2.csv',root_dir='data/testdata')
        test_dataset_3 = FeTa_data('axis3.csv',root_dir='data/testdata')

        # VOLUMES
        I1, S1 = np.zeros((256, 256, 256)), np.zeros((256, 256, 256))
        I2, S2 = np.zeros((256, 256, 256)), np.zeros((256, 256, 256))
        I3, S3 = np.zeros((256, 256, 256)), np.zeros((256, 256, 256))
        
        avg_dice =[]
        avg_class_dice = np.zeros(7)
            
        if args.mode=='demo':
            patients =['27']
        else:
            patients = test_dataset_1.get_patients()

        for patient in patients:
            print(f'Processing patient {patient}')
            for i in range(0,255):
            
                image1,label1,name1 = test_dataset_1.__getitem__(patient,i)
                image2,label2,name2 = test_dataset_2.__getitem__(patient,i)
                image3,label3,name3 = test_dataset_3.__getitem__(patient,i)

                I1[:, :, i] = image1
                I2[:, i, :] = image2
                I3[i, :, :] = image3
                S1[:, :, i] = label1
                S2[:, i, :] = label2
                S3[i, :, :] = label3

            v1=test.evaluate(net1, I1, S1,device, 1)
            v2=test.evaluate(net2, I2, S2,device, 2)
            v3=test.evaluate(net3, I3, S3,device, 3)

            VolTotal = (v1+v2+v3)/3

            # TEST
            print('Testing...')
            if args.mode == 'demo':
                mean_dice, class_dice = test.test(VolTotal,S1,demo=True)
            else:
                mean_dice, class_dice = test.test(VolTotal,S1)
            avg_dice.append(mean_dice)
            avg_class_dice+=class_dice

        # CALCULATE THE FINAL METRICS
        avg_Dice = np.mean(np.array(avg_dice))
        avg_TissueDice = avg_class_dice/len(patients)
        print(f'Average Dice Score: {avg_Dice}')
        print(f'Average Tissue Dice Score: {avg_TissueDice}')

    else:
        raise ValueError('Mode must be either unet2D or unet25D')

    print('done')

    return

if __name__ == '__main__':

    #---------UNET---------------
    # SET THE PARAMETERS
    parser = argparse.ArgumentParser()
    # EXPERIMENT DETAILS
    parser.add_argument('--mode', type=str, default='test',choices = ['demo', 'test'],
                        help='mode to test/demo (default: demo)')
    parser.add_argument('--experiment', type=str, default='unet25D',choices = ['unet2D','unet25D','ROG'],
                        help='mode to test/demo (default: demo)')
    parser.add_argument('--nameUnet', type=str, default='unet2.5D',
                        help='Name of the current experiment (default: unet2.5D)')
    parser.add_argument('--testUnet', action='store_false', default=True,
                        help='Evaluate a model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU(s) to use (default: 0)')
    args = parser.parse_args()
    

    main(args)


