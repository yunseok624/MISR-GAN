"""
Adapted from: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/models/esrgan_model.py
Authors: XPixelGroup
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import wandb

from basicsr.archs import build_network
from basicsr.models.srgan_model import SRGANModel
from basicsr.models.sr_model import SRModel
from basicsr.utils import USMSharp, get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import master_only
from basicsr.utils.registry import MODEL_REGISTRY

from ssr.losses import build_loss
from ssr.metrics import calculate_metric

from kornia.losses import MS_SSIMLoss as ms_ssim

def three_grayscale(img):
    img_list = [img.clone() for _ in range(3)]
    three = torch.cat(img_list, dim=1)
    return three

def normalize(img):
    img_norm = 2*((img - torch.min(img)) / (torch.max(img) - torch.min(img))) - 1
    return img_norm

@MODEL_REGISTRY.register()
class SSRESRGANModel(SRGANModel):
    """
    SSR ESRGAN Model: Satellite imagery super-resolution model based on the Real-ESRGAN architecture.

    The input to the generator is a time series of low-res images, and it learns to generate
    a higher resolution image. The discriminator then sees the generated images and GT images,
    with the optional additional input of an older high-res image and the same low-res images
    that the generator receives.
    """

    def __init__(self, opt):
        super(SSRESRGANModel, self).__init__(opt)
        self.usm_sharpener = USMSharp().cuda()

        self.scale = opt['scale']

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
#         self.setup_schedulers()

    @torch.no_grad()
    def feed_data(self, data):
        if len(data['lr'][0, :, 0, 0]) <= 9:
            self.lr = data['lr'].to(self.device).float()
        else:
            self.lr = data['lr'].to(self.device).float()/255
        if 'hr' in data:
            if len(data['hr'][0, :, 0, 0]) == 1:
                self.gt = data['hr'].to(self.device).float()
            else:
                self.gt = data['hr'].to(self.device).float()/255
            self.gt_usm = self.usm_sharpener(self.gt)  # sharpen the ground truth as in Real-ESRGAN paper

        # Provide a high-res image at the same location, but different timestamp from the gt, to the discriminator.
        self.old_hr = None
        if 'old_hr' in data:
            self.old_hr = data['old_hr'].to(self.device).float()/255

        # Feed discriminator the same low-res images that the generator receives.
        self.feed_disc_lr = True if ('feed_disc_lr' in self.opt and self.opt['feed_disc_lr']) else False

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # Upsample the low-res input images to that of the ground truth so they can be stacked.
        lr_shp = self.lr.shape
        lr_resized = F.interpolate(self.lr, scale_factor=self.scale)

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lr)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
        
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(l1_gt, self.output)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # perceptual loss
            if self.cri_perceptual:
                if len(self.output[0, :, 0, 0]) == 1:
#                     l_g_percep, _ = self.cri_perceptual(three_grayscale(self.output), three_grayscale(percep_gt))
                    l_g_percep = self.cri_perceptual(normalize(three_grayscale(self.output)), normalize(three_grayscale(percep_gt)))
                else:
#                     l_g_percep, _ = self.cri_perceptual(self.output, percep_gt)
                    l_g_percep = self.cri_perceptual(normalize(self.output), normalize(percep_gt))
                
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep

            # Stack additional information onto the Real/Fake image given to the discriminator.
            # Specifically an older high-res image corresponding to the location of the ground truth,
            # and/or the same low-res images that were fed to the generator for this datapoint.
            if (self.old_hr is not None) and self.feed_disc_lr:
                real_disc_input = torch.cat((gan_gt, lr_resized, self.old_hr), dim=1)
                fake_disc_input = torch.cat((self.output, lr_resized, self.old_hr), dim=1)
            elif (self.old_hr is not None):
                real_disc_input = torch.cat((gan_gt, self.old_hr), dim=1)
                fake_disc_input = torch.cat((self.output, self.old_hr), dim=1)
            elif self.feed_disc_lr:
                real_disc_input = torch.cat((gan_gt, lr_resized), dim=1)
                fake_disc_input = torch.cat((self.output, lr_resized), dim=1)
            else:
                real_disc_input = gan_gt
                fake_disc_input = self.output

            # gan loss (relativistic gan)
            real_g_pred = self.net_d(real_disc_input.detach())
            fake_g_pred = self.net_d(fake_disc_input)
#             l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_real = self.cri_gan(real_g_pred - torch.mean(fake_g_pred), False, is_disc=False) * 0.5
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_g_pred), True, is_disc=False) * 0.5
            l_g_gan = l_g_real + l_g_fake
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            
            l_g_total.backward()
            loss_dict['gen_loss'] = l_g_total
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        # Stack additional information onto the Real/Fake image given to the discriminator.
        # Specifically an older high-res image corresponding to the location of the ground truth,
        # and/or the same low-res images that were fed to the generator for this datapoint.
        if (self.old_hr is not None) and self.feed_disc_lr:
            real_disc_input = torch.cat((gan_gt, lr_resized, self.old_hr), dim=1)
            fake_disc_input = torch.cat((self.output, lr_resized, self.old_hr), dim=1)
        elif self.old_hr is not None:
            real_disc_input = torch.cat((gan_gt, self.old_hr), dim=1)
            fake_disc_input = torch.cat((self.output, self.old_hr), dim=1)
        elif self.feed_disc_lr:
            fake_disc_input = torch.cat((self.output, lr_resized), dim=1)
            real_disc_input = torch.cat((gan_gt, lr_resized), dim=1)
        else:
            fake_disc_input = self.output
            real_disc_input = gan_gt

        self.optimizer_d.zero_grad()
        
        # gan loss (relativistic gan)
        # real
        real_d_pred = self.net_d(real_disc_input)
        fake_d_pred = self.net_d(fake_disc_input.detach())
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward(retain_graph=True)
        
        # fake
        fake_d_pred = self.net_d(fake_disc_input.detach())
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred), False, is_disc=True) * 0.5
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        
        loss_dict['dis_loss'] = l_d_real + l_d_fake
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lr)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lr)
            self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lr'] = self.lr.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def _initialize_best_metric_results(self, dataset_name, metrics2run):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in metrics2run.items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def validation(self, dataloader, current_iter, tb_logger, save_img):
        
        dataset_name = dataloader.dataset.opt['name']

        with_metrics = False
        if dataset_name == 'test':
            with_metrics = self.opt['test'].get('metrics') is not None
            if with_metrics:
                metrics2run = self.opt['test']['metrics']
        else:
            with_metrics = self.opt['val'].get('metrics') is not None
            if with_metrics:
                metrics2run = self.opt['val']['metrics']
        
        if dataset_name == 'test':
            use_pbar = self.opt['test'].get('pbar', False)
        else:
            use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in metrics2run.keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name, metrics2run)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # TODO: the savename logic below does not work for val batch size > 1
            img_name = str(idx)

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            
            if dataset_name == 'test':
                lr_img = tensor2img(visuals['lr'][:, 0:1, :, :])
                metric_data['lr'] = lr_img

            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                
                if save_img:
                    if self.opt['is_train']:
                        save_img_path = os.path.join(self.opt['path']['visualization'], img_name,
                                                     f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = os.path.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}_gt.png')
                    imwrite(gt_img, save_img_path)
                del self.gt

            # tentative for out of GPU memory
            del self.lr
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = os.path.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = os.path.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                
                imwrite(sr_img, save_img_path)
                                                 
            if save_img:
                if dataset_name == 'test':
                    save_img_path = os.path.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}_lr.png')
                imwrite(lr_img, save_img_path)
            
            if with_metrics:
                # calculate metrics
                for name, opt_ in metrics2run.items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(metric_data['img'])
        ax1.set_title('Super Resolved')
        ax2.imshow(metric_data['img2'])
        ax2.set_title('Ground Truth')
        
        tb_logger.add_figure('Training result', fig, current_iter)

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth.tar'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                print("Saving model weights to...", save_path)
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

###############################################################################################################################
###############################################################################################################################

@MODEL_REGISTRY.register()
class SSRESRGANModel_PSNR(SRModel):
    """
    SSR ESRGAN Model: Satellite imagery super-resolution model based on the Real-ESRGAN architecture.

    The input to the generator is a time series of low-res images, and it learns to generate
    a higher resolution image. The discriminator then sees the generated images and GT images,
    with the optional additional input of an older high-res image and the same low-res images
    that the generator receives.
    """

    def __init__(self, opt):
        super(SSRESRGANModel_PSNR, self).__init__(opt)
        self.usm_sharpener = USMSharp().cuda()

        self.scale = opt['scale']

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        self.net_g.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # set up optimizers and schedulers
        self.setup_optimizers()
#         self.setup_schedulers()

    @torch.no_grad()
    def feed_data(self, data):
        if len(data['lr'][0, :, 0, 0]) <= 9:
            self.lr = data['lr'].to(self.device).float()
        else:
            self.lr = data['lr'].to(self.device).float()/255
        if 'hr' in data:
            if len(data['hr'][0, :, 0, 0]) == 1:
                self.gt = data['hr'].to(self.device).float()
            else:
                self.gt = data['hr'].to(self.device).float()/255
            self.gt_usm = self.usm_sharpener(self.gt)  # sharpen the ground truth as in Real-ESRGAN paper

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt

        # optimize net_g
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lr)

        l_g_total = 0
        loss_dict = OrderedDict()
        
        # pixel loss
        if self.cri_pix:
            l_g_pix = self.cri_pix(l1_gt, self.output)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

        l_g_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lr)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lr)
            self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lr'] = self.lr.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def _initialize_best_metric_results(self, dataset_name, metrics2run):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in metrics2run.items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']

        with_metrics = False
        if dataset_name == 'test':
            with_metrics = self.opt['test'].get('metrics') is not None
            if with_metrics:
                metrics2run = self.opt['test']['metrics']
        else:
            with_metrics = self.opt['val'].get('metrics') is not None
            if with_metrics:
                metrics2run = self.opt['val']['metrics']
        
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in metrics2run.keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name, metrics2run)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # TODO: the savename logic below does not work for val batch size > 1
            img_name = str(idx)

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                
                if save_img:
                    if self.opt['is_train']:
                        save_img_path = os.path.join(self.opt['path']['visualization'], img_name,
                                                     f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = os.path.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}_gt.png')
                    imwrite(gt_img, save_img_path)
                del self.gt

            # tentative for out of GPU memory
            del self.lr
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = os.path.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = os.path.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                
                imwrite(sr_img, save_img_path)
            
            if with_metrics:
                # calculate metrics
                for name, opt_ in metrics2run.items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(metric_data['img'])
        ax1.set_title('Super Resolved')
        ax2.imshow(metric_data['img2'])
        ax2.set_title('Ground Truth')
        
        tb_logger.add_figure('Training result', fig, current_iter)

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                print("Saving model weights to...", save_path)
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')