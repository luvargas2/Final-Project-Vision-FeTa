from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.data_loading import BasicDataset, CarvanaDataset
from unet import UNet
import os
from PIL import Image
import numpy as np
import pickle
import glob
import cv2
from sklearn.metrics import jaccard_score
from torch import Tensor

def test(gpu=0,demo=False):
    def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon=1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        if input.dim() == 2 and reduce_batch_first:
            raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

        if input.dim() == 2 or reduce_batch_first:
            inter = torch.dot(input.reshape(-1), target.reshape(-1))
            sets_sum = torch.sum(input) + torch.sum(target)
            if sets_sum.item() == 0:
                sets_sum = 2 * inter

            return (2 * inter + epsilon) / (sets_sum + epsilon)
        else:
            # compute and average metric for each batch element
            dice = 0
            for i in range(input.shape[0]):
                dice += dice_coeff(input[i, ...], target[i, ...])
            return dice / input.shape[0]


    def multiclass_dice_coeff(input, target, reduce_batch_first: bool = False, epsilon=1e-6):
        # Average of Dice coefficient for all classes
        assert input.size() == target.size()
        dice = 0
        dice_class = []
        for channel in range(input.shape[1]):
            d = dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
            dice += d
            dice_class.append(d)
        return dice / input.shape[1], dice_class


    def dimensiones(I):
        x, y, z = I.shape
        im = np.zeros((y, z, x))
        for i in range(x):
            im[:, :, i] = I[i, :, :]
        return im


    def seg_pred(S):
        z, x, y = S.shape
        s = np.zeros((x, y))
        for i in range(x):
            for j in range(y):
                m = np.copy(S[:, i, j])
                m.sort()
                m = m[-1]
                s[i, j] = np.where(S[:, i, j] == m)[0]
        return s


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


    def load(filename):
        return Image.open(filename)



    def evaluate(net, dataloader, mask, device):
        net.eval()
        dice_score = 0
        # iterate over the validation set
        image=dataloader

        image = cv2.resize(image, (64, 64))

        image = torch.as_tensor(np.array([[image]])).float()

        image = image.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_pred=cv2.resize(dimensiones(mask_pred.cpu().numpy()[0, :, :, :]), (128, 128))

        net.train()

        return mask_pred


    def validar(Validacion):
        imagen1 = glob.glob(os.path.join('data','testpatches','test1cortes','imgs', f'{Validacion}', '*.png'))
        seg1 = glob.glob(os.path.join('data','testpatches','test1cortes','masks', f'{Validacion}', '*.png'))
        
        volumen=np.zeros((256,256,256,8))
        S1=np.zeros((256,256,256))

        net1 = UNet(n_channels=1, n_classes=8, bilinear=False)
        device = torch.device('cuda:%s' % gpu)
        net1.load_state_dict(torch.load('models/unet2Dalbcortes24.pth'))
        net1.to(device=device)

        matriz_pesos=np.ones((256,256,256,8))
        matriz_pesos[:,:128, 64:192, :] = matriz_pesos[:,:128, 64:192, :] / 2
        matriz_pesos[:,64:192, :128, :] = matriz_pesos[:,64:192, :128, :] / 2
        matriz_pesos[:,64:192, 64:192, :] = matriz_pesos[:,64:192, 64:192, :] / 2
        matriz_pesos[:,64:192, 128:, :] = matriz_pesos[:,64:192, 128:, :] / 2
        matriz_pesos[:,128:, 64:192, :] = matriz_pesos[:,128:, 64:192, :] / 2
        for i in tqdm(range(len(imagen1))):
            r_im=[]
            r_seg=[]
            l_seg1 = imagen1[i].split('/')[-1].split('.')[0].split('-')[-2]
            l3_seg1_im = imagen1[i].split('/')[-1].split('.')[0].split('-')[-1]
            for j in range(len(seg1)):
                l2_seg1 = seg1[j].split('/')[-1].split('.')[0].split('-')[-2]#.split('_')[0]
                l3_seg1 = seg1[j].split('/')[-1].split('.')[0].split('-')[-1]#.split('_')[0]
                if l_seg1==l2_seg1 and l3_seg1 == l3_seg1_im:
                    r_im = preprocess(load(imagen1[i]), 1, 0)
                    r_seg = preprocess(load(seg1[j]), 1, 1)
                    break
            v1 = evaluate(net1, r_im[0], r_seg, device)
            x,y=divmod(int(l3_seg1),3)
            x=x*64
            y=y*64
            volumen[int(l_seg1),x:x+128,y:y+128,:]+=v1
            S1[int(l_seg1),x:x+128,y:y+128]=r_seg

        volumen=volumen*matriz_pesos

        dice_score=0
        tissues=np.zeros(7)

        for i in tqdm(range(volumen.shape[0])):

            pred = volumen[i, :, :, :]
            pred = torch.as_tensor(np.array([pred]))
            gt = S1[i, :, :]
            gt = torch.as_tensor(np.array([gt]), dtype=torch.long)
            mask_true = F.one_hot(gt, 8).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(pred.argmax(dim=3), 8).permute(0, 3, 1, 2).float()
            m = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                    reduce_batch_first=False)
            dice_score += m[0].item()
            tissues += m[1]

        print(f'Average Dice Score: {dice_score / volumen.shape[2]}')
        print(f'Average Tissue Dice Score: {tissues/volumen.shape[2]}')

        return dice_score / volumen.shape[2]

    #Val=[77,70,75,37,38,52,5,1,29,20]

   
    if demo:
        print('Running demo for patient 27')
        Test = [27]
    else:
        print('Running test')
        Test = [16,23,24,27,28,32,35,36,59,64]


    D=[]
    for i in Test:
        v=validar(i)
        D.append(v)
    print(np.mean(np.array(D)))

