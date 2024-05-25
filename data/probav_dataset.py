import os
import cv2
import matplotlib.pyplot as plt
import glob
import torch
import random
import torchvision.transforms as transforms
import numpy as np
from torch.utils import data as data
import heapq
import albumentations as A
from albumentations.pytorch import ToTensorV2

from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class PROBAVDataset(data.Dataset):
    """
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
    """

    def __init__(self, opt):
        super(PROBAVDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        self.data_root = opt['data_root']

        self.n_lr_images = opt['n_lr_images']
        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

#         hr_fps = glob.glob(self.data_root + 'train/NIR/*/HR.png')
        
        # Filter filepaths based on if the split is train or validation.
        if self.split == 'train':
            hr_fps = glob.glob(self.data_root + '/*/HR.png')
        else:
            hr_fps = glob.glob(self.data_root + '/*/HR.png')

        self.datapoints = []
        for hr_fp in hr_fps:
            temp = []
            directory, _ = os.path.split(hr_fp)
            for i in range(len(glob.glob(os.path.join(directory, 'LR*.png')))):
                if i < 10:
                    qm_temp = hr_fp.replace('HR', 'QM00' + str(i))
                else:
                    qm_temp = hr_fp.replace('HR', 'QM0' + str(i))
                temp.append(qm_temp)
            idx_names = np.array([os.path.basename(t)[-7:-4] for t in temp])
            cl_npy = hr_fp.replace('HR.png', 'clearance.npy')
            clearance = np.load(cl_npy)
            top = heapq.nlargest(self.n_lr_images, range(len(clearance)), clearance.take)
            idx_names = idx_names[top]
            folder_name = os.path.basename(os.path.dirname(hr_fp))

            lrs = [os.path.join(directory, f'LR{i}.png') for i in idx_names]
            self.datapoints.append([hr_fp, lrs, folder_name])

        self.data_len = len(self.datapoints)
        print("Loaded ", self.data_len, " data pairs for split ", self.split)
        
#         self.transform = A.Compose([
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             ToTensorV2()],
#             additional_targets={'image0': 'image', 'image1': 'image', 'image2': 'image', 
#                                 'image3': 'image', 'image4': 'image', 'image5': 'image',
#                                 'image6': 'image', 'image7': 'image', 'image8': 'image'},
#             is_check_shapes=False
#         )
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()],
            additional_targets={f'image{i}': 'image' for i in range(self.n_lr_images)},
            is_check_shapes=False
        )

    def __getitem__(self, index):
        hr_path, lr_paths, names  = self.datapoints[index]

        if self.split == 'train':
            hr_im = plt.imread(hr_path)

            lr_ims = []
            for lr_path in lr_paths:
                lr_im = plt.imread(lr_path)
                lr_ims.append(lr_im)

#             transformed = self.transform(image=hr_im, image0=lr_ims[0], image1=lr_ims[1], image2=lr_ims[2],
#                                          image3=lr_ims[3], image4=lr_ims[4], image5=lr_ims[5], image6=lr_ims[6], 
#                                          image7=lr_ims[7], image8=lr_ims[8])
            transformed = self.transform(image=hr_im, **{f'image{i}': lr_ims[i] for i in range(self.n_lr_images)})
            img_HR = transformed['image']
            
#             img_LR_list = []
#             for i in range(0, 9):
#                 lr_im_new = transformed[f'image{i}']
#                 img_LR_list.append(lr_im_new)
            img_LR_list = [transformed[f'image{i}'] for i in range(self.n_lr_images)]
            img_LR = np.concatenate(img_LR_list)
        else:
            hr_im = plt.imread(hr_path)

            hr_tensor = torch.tensor(hr_im)

            lr_ims = []
            for lr_path in lr_paths:
                lr_im = plt.imread(lr_path)
                lr_tensor = torch.tensor(lr_im)
                lr_ims.append(lr_tensor.unsqueeze(0))

            if self.use_3d:
                img_LR = torch.stack(lr_ims)
            else:
                img_LR = torch.cat(lr_ims)

            img_HR = hr_tensor.unsqueeze(0)

        return {'hr': img_HR, 'lr': img_LR, 'Index': index, 'name': names}

    def __len__(self):
        return self.data_len