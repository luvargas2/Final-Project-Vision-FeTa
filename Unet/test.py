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


def multiclass_dice_coeff(input, target, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    dice_class=[]
    for channel in range(input.shape[1]):
        d=dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        dice += d
        dice_class.append(d)
    return dice / input.shape[1],dice_class

def dimensiones(I):
    x, y, z = I.shape
    im = np.zeros((y, z, x))
    for i in range(x):
        im[:, :, i] = I[i, :, :]
    return im

def create_matrix(): 
    matriz_pesos=np.ones((256,256,256,8))
    matriz_pesos[:,:128, 64:192, :] = matriz_pesos[:,:128, 64:192, :] / 2
    matriz_pesos[:,64:192, :128, :] = matriz_pesos[:,64:192, :128, :] / 2
    matriz_pesos[:,64:192, 64:192, :] = matriz_pesos[:,64:192, 64:192, :] / 2
    matriz_pesos[:,64:192, 128:, :] = matriz_pesos[:,64:192, 128:, :] / 2
    matriz_pesos[:,128:, 64:192, :] = matriz_pesos[:,128:, 64:192, :] / 2
    return matriz_pesos

def evaluate(net, img, mask, device, axis):
    print(f'Evaluating axis {axis}... ')
    net.eval()
    dice_score = 0
    volumen = np.zeros((256, 256, 256, 8))
    # iterate over the validation set
    for i in tqdm(range(256)):
        if axis == 1:
            image, mask_true = img[:, :, i], mask[:, :, i]
        elif axis == 2:
            image, mask_true = img[:, i, :], mask[:, i, :]
        elif axis == 3:
            image, mask_true = img[i, :, :], mask[i, :, :]
        image = cv2.resize(image, (128, 128))
        # move images and labels to correct device and type
        image = torch.as_tensor(np.array([[image]])).float()
        mask_true = torch.as_tensor(np.array([[mask_true]])).long()
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes)  # .permute(0, 3, 1, 2).float()
        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            if axis == 1:
                v = np.zeros((256, 256, 8))
                for j in range(8):
                    v[:, :, j] = cv2.resize(dimensiones(mask_pred.cpu().numpy()[0, :, :, :])[:, :, j], (256, 256))
                volumen[:, :, i, :] = v
            elif axis == 2:
                v = np.zeros((256, 256, 8))
                for j in range(8):
                    v[:, :, j] = cv2.resize(dimensiones(mask_pred.cpu().numpy()[0, :, :, :])[:, :, j], (256, 256))
                volumen[:, i, :, :] = v
            elif axis == 3:
                v = np.zeros((256, 256, 8))
                for j in range(8):
                    v[:, :, j] = cv2.resize(dimensiones(mask_pred.cpu().numpy()[0, :, :, :])[:, :, j], (256, 256))
                volumen[i, :, :, :] = v
    net.train()

    return volumen


def test(volumen,mask):

    dice_score=0
    tissues=np.zeros(7)
    for i in tqdm(range(volumen.shape[2])):
        pred=volumen[:,:,i,:]
        pred=torch.as_tensor(np.array([pred]))
        gt=mask[:,:,i]
        gt=torch.as_tensor(np.array([gt]),dtype=torch.long)
        mask_true = F.one_hot(gt, 8).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(pred.argmax(dim=3), 8).permute(0, 3, 1, 2).float()
        m=multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                            reduce_batch_first=False)
        dice_score += m[0]
        tissues+=m[1]
    mean_dice = dice_score/volumen.shape[2]
    class_dice = tissues/volumen.shape[2]

    return mean_dice, class_dice

def evaluate_patch(net, dataloader, mask, device):
    net.eval()
    dice_score = 0
    # iterate over the validation set
    image=dataloader

    image = cv2.resize(image, (64, 64))
    # move images and labels to correct device and type
    image = torch.as_tensor(np.array([[image]])).float()
    image = image.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # predict the mask
        mask_pred = net(image)
        mask_pred=cv2.resize(dimensiones(mask_pred.cpu().numpy()[0, :, :, :]), (128, 128))

    net.train()
    
    return mask_pred

