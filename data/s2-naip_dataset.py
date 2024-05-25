import os
import json
import glob
import torch
import random
import torchvision
import torchvision.transforms as transforms
import skimage.measure
import skimage
import numpy as np
import torch.nn.functional as F
from torch.utils import data as data
from torch.utils.data import WeightedRandomSampler

from basicsr.utils.registry import DATASET_REGISTRY

from ssr.utils.data_utils import has_black_pixels, has_white_pixels, has_cloud_pixels

random.seed(123)

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler except allows for more than 2^24 samples to be sampled.
    Source code: https://github.com/pytorch/pytorch/issues/2576#issuecomment-831780307
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

@DATASET_REGISTRY.register()
class S2NAIPDataset(data.Dataset):
    """
    Dataset object for the S2NAIP data. Builds a list of Sentinel-2 time series and NAIP image pairs.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            sentinel2_path (str): Data path for Sentinel-2 imagery.
            naip_path (str): Data path for NAIP imagery.
            n_sentinel2_images (int): Number of Sentinel-2 images to use as input to model.
            scale (int): Upsample amount, only 4x is supported currently.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(S2NAIPDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        train = True if self.split == 'train' else False

        # Random cropping and resizing augmentation; training only
        self.rand_crop = opt['rand_crop'] if 'rand_crop' in opt else False

        self.n_s2_images = int(opt['n_s2_images'])
        self.scale = int(opt['scale'])

        # Flags whether the model being used expects [b, n_images, channels, h, w] or [b, n_images*channels, h, w].
        # The L2-based models expect the first shape, while the ESRGAN models expect the latter.
        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

        # Path to high-res images of older timestamps and corresponding locations to training data.
        # In the case of the S2NAIP dataset, that means NAIP images from 2016-2018.
        self.old_naip_path = opt['old_naip_path'] if 'old_naip_path' in opt else None

        # Path to osm_chips_to_masks.json if provided. 
        self.osm_chips_to_masks = opt['osm_objs_path'] if 'osm_objs_path' in opt else None

        # Sentinel-2 bands to be used as input. Default to just using tci.
        self.s2_bands = opt['s2_bands'] if 's2_bands' in opt else ['tci']
        # Move tci to front of list for later logic.
        self.s2_bands.insert(0, self.s2_bands.pop(self.s2_bands.index('tci')))

        # If a path to older NAIP imagery is provided, build dictionary of each chip:path to png.
        if self.old_naip_path is not None:
            old_naip_chips = {}
            for old_naip in glob.glob(self.old_naip_path + '/**/*.png', recursive=True):
                old_chip = old_naip.split('/')[-1][:-4]

                if not old_chip in old_naip_chips:
                    old_naip_chips[old_chip] = []
                old_naip_chips[old_chip].append(old_naip)

        # If a path to osm_chips_to_masks.json is provided, we want to filter out datapoints where
        # there is not at least n_osm_objs objects in the NAIP image.
        if self.osm_chips_to_masks is not None and train:
            osm_obj_data = json.load(open(self.osm_chips_to_masks))
            print("Loaded osm_chip_to_masks.json with ", len(osm_obj_data), " entries.")

        # Paths to Sentinel-2 and NAIP imagery.
        self.s2_path = opt['sentinel2_path']
        self.naip_path = opt['naip_path']
        if not (os.path.exists(self.s2_path) and os.path.exists(self.naip_path)):
            raise Exception("Please make sure the paths to the data directories are correct.")

        self.naip_chips = glob.glob(self.naip_path + '/**/*.png', recursive=True)

        # Reduce the training set down to a specified number of samples. If not specified, whole set is used.
        if 'train_samples' in opt and train:
            self.naip_chips = random.sample(self.naip_chips, opt['train_samples'])

        self.datapoints = []
        for n in self.naip_chips:
            # Extract the X,Y chip from this NAIP image filepath.
            split_path = n.split('/')
            chip = split_path[-2]

            # If old_hr_path is specified, grab an old high-res image (NAIP) for the current datapoint.
            if self.old_naip_path is not None:
                old_chip = old_naip_chips[chip][0]

            # If using OSM Object ESRGAN, filter dataset to only include images containing OpenStreetMap objects.
            if self.osm_chips_to_masks is not None and train:
                if not (chip in osm_obj_data and sum([len(osm_obj_data[chip][k]) for k in osm_obj_data[chip].keys()]) >= opt['n_osm_objs']):
                    continue

            # Gather the filepaths to the Sentinel-2 bands specified in the config.
            s2_paths = [os.path.join(self.s2_path, chip, band + '.png') for band in self.s2_bands]

            # Return the low-res, high-res, chip (ex. 12345_67890), and [optionally] older high-res image paths. 
            if self.old_naip_path:
                self.datapoints.append([n, s2_paths, chip, old_chip])
            else:
                self.datapoints.append([n, s2_paths, chip])

        self.data_len = len(self.datapoints)
        print("Number of datapoints for split ", self.split, ": ", self.data_len)

    def get_tile_weight_sampler(self, tile_weights):
        weights = []
        for dp in self.datapoints:
            # Extract the NAIP chip from this datapoint's NAIP path.
            # With the chip, we can index into the tile_weights dict (naip_chip : weight)
            # and then weight this datapoint pair in self.datapoints based on that value.
            naip_path = dp[0]
            split = naip_path.split('/')[-1]
            chip = split[:-4]

            # If the chip isn't in the tile weights dict, then there weren't any OSM features
            # in that chip, so we can set the weight to be relatively low (ex. 1).
            if not chip in tile_weights:
                weights.append(1)
            else:
                weights.append(tile_weights[chip])

        print('Using tile_weight_sampler, min={} max={} mean={}'.format(min(weights), max(weights), np.mean(weights)))
        return CustomWeightedRandomSampler(weights, len(self.datapoints))

    def __getitem__(self, index):

        # A while loop and try/excepts to catch a few images that we want to ignore during 
        # training but do not necessarily want to remove from the dataset, such as the
        # ground truth NAIP image being partially invalid (all black). 
        counter = 0
        while True:
            index += counter  # increment the index based on what errors have been caught
            if index >= self.data_len:
                index = 0

            datapoint = self.datapoints[index]

            if self.old_naip_path:
                naip_path, s2_paths, zoom17_tile, old_naip_path = datapoint[0], datapoint[1], datapoint[2], datapoint[3]
            else:
                naip_path, s2_paths, zoom17_tile = datapoint[0], datapoint[1], datapoint[2]

            # Load the 128x128 NAIP chip in as a tensor of shape [channels, height, width].
            naip_chip = torchvision.io.read_image(naip_path)

            # Check for black pixels (almost certainly invalid) and skip if found.
            if has_black_pixels(naip_chip) or has_white_pixels(naip_chip):
                counter += 1
                continue
            img_HR = naip_chip

            # Load the T*32x32xC S2 files for each band in as a tensor.
            # There are a few rare cases where loading the Sentinel-2 image fails, skip if found.
            try:
                s2_tensor = None
                for i, s2_path in enumerate(s2_paths):

                    # There are tiles where certain bands aren't available, use zero tensors in this case.
                    if not os.path.exists(s2_path):
                        img_size = (self.n_s2_images, 3, 32, 32) if 'tci' in s2_path else (self.n_s2_images, 1, 32, 32)
                        s2_img = torch.zeros(img_size, dtype=torch.uint8)
                    else:
                        s2_img = torchvision.io.read_image(s2_path)
#                         s2_img = plt.imread(s2_path).transpose(2, 0, 1)
                        s2_img = torch.reshape(s2_img, (-1, s2_img.shape[0], 32, 32))
#                         x1 = torch.from_numpy(s2_img[0:32, :, :]).permute(2, 0, 1)
#                         for i in range(1, len(s2_naip[:, 0, 0])//32):
#                             if i == 1:
#                                 x2 = torch.from_numpy(s2_naip[i*32:(i+1)*32, :, :]).permute(2, 0, 1)
#                                 x3 = torch.cat([x1, x2], axis=0)
#                                 x3 = x3.reshape(i*2, 3, 32, 32)
#                                 x1 = x3
#                             else:
#                                 x2 = torch.from_numpy(s2_naip[(i-1)*32:i*32, :, :]).permute(2, 0, 1)
#                                 x2 = x2.unsqueeze(0)
#                                 x3 = torch.cat([x1, x2], axis=0)
#                                 x3 = x3.reshape(i*2, 3, 32, 32)
#                                 x1 = x3
#                         s2_img_new = x3

                    if i == 0:
                        s2_tensor = s2_img
                    else:
                        s2_tensor = torch.cat((s2_tensor, s2_img), dim=1)
            except:
                counter += 1
                continue

            # Skip the cases when there are not as many Sentinel-2 images as requested.
            if s2_tensor.shape[0] < self.n_s2_images:
                counter += 1
                continue

            # Iterate through the 32x32 tci chunks at each timestep, separating them into "good" (valid)
            # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
            tci_chunks = s2_tensor[:, :3, :, :]
            goods, bads = [], []
            for i, ts in enumerate(tci_chunks):
                if has_black_pixels(ts) or has_white_pixels(ts) or has_cloud_pixels(ts):
                    bads.append(i)
                else:
                    goods.append(i)
#             mean = []
#             rand_indices = []

#             for i,ts in enumerate(tci_chunks):
#                 img_np = ts.detach().cpu().numpy()
#                 img_np = img_np.transpose(1, 2, 0)
#                 m = img_np.mean()
#                 mean.append([i, ts, img_np, m])
                
#                 mean.sort(key=lambda x: (x[3]))
#                 top_mean = mean[-8:]
            
            # Pick self.n_s2_images random indices of S2 images to use. Skip ones that are partially black.
            if len(goods) >= self.n_s2_images:
                rand_indices = random.sample(goods, self.n_s2_images)
            else:
                need = self.n_s2_images - len(goods)
                rand_indices = goods + random.sample(bads, need)
#             for _, _, _, ent in top_mean:
#                 rand_indices.append(j)
            rand_indices_tensor = torch.as_tensor(rand_indices)

            # Extract the self.n_s2_images from the first dimension.
            img_S2 = s2_tensor[rand_indices_tensor]

            # If the rand_crop augmentation is specified (during training only), randomly pick size in [24,32]
            # and randomly crop the LR and HR images to their respective sizes, then resize back to 32x32 / 128x128.
            if self.rand_crop:
                rand_lr_size = random.randint(24, 32)
                rand_hr_size = int(rand_lr_size * 4)
                img_S2_cropped = img_S2[:, :, :rand_lr_size, :rand_lr_size]
                img_HR_cropped = img_HR[:, :rand_hr_size, :rand_hr_size]
                img_S2 = F.interpolate(img_S2_cropped, (32,32))
                img_HR = F.interpolate(img_HR_cropped.unsqueeze(0), (128,128)).squeeze(0)  # need to unsqueeze tensor for interpolation to work, then squeeze

            # If using a model that expects 5 dimensions, we will not reshape to 4 dimensions.
            if not self.use_3d:
                img_S2 = torch.reshape(img_S2, (-1, 32, 32))
            
            if self.old_naip_path is not None:
                old_naip_chip = torchvision.io.read_image(old_naip_path)
#                 old_naip_chip = plt.imread(old_naip_path).transpose(2, 0, 1)
#                 old_naip_chip_new = torch.from_numpy(old_naip_chip)
                img_old_HR = old_naip_chip
                return {'hr': img_HR, 'lr': img_S2, 'old_hr': img_old_HR, 'Index': index, 'Phase': self.split, 'Chip': zoom17_tile}

            return {'hr': img_HR, 'lr': img_S2, 'Index': index, 'Phase': self.split, 'Chip': zoom17_tile}

    def __len__(self):
        return self.data_len
