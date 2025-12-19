#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on  2024


"""
import os
import h5py
import numpy as np
import torch
import glob
import pickle
import sklearn

from fastmri.data.subsample import RandomMaskFunc, EquispacedMaskFunc
from fastmri.pl_modules.varnet_module import VarNetModuleMidc, VarNetModule
import fastmri.data.transforms as T
import fastmri
from fastmri.models import MocoSimLayer
from tqdm import tqdm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def ssim(gt, pred):
    return structural_similarity(gt,pred, multichannel=True, data_range=gt.max())

def nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def psnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())

def compute_scores(gt, pred):
    return [ssim(gt, pred), nmse(gt, pred), psnr(gt, pred)]

if __name__ == '__main__':
    noMotionFrac_list = [0.0, 1.1]
    filenames_list = ['./scratch/val_varnet_varnetMidc_svm_withMotion', './scratch/val_varnet_varnetMidc_svm_noMotion']
    for indx in range(2):
        # ---- load model midc ---- #
        H, W = 320, 320
        mocoSimUnd = MocoSimLayer(noMotionFrac=noMotionFrac_list[indx], half_width_no_motion=8.0) # motion simulation layer
        mask_func = EquispacedMaskFunc(center_fractions=[0.08], accelerations=[4]) # under sampling layer
        model_mc = VarNetModuleMidc(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8)
        path_to_checkpoint = 'D:\DeepLearning_MOCO\training_epoches\epoch=32.ckpt'
        checkpoint = torch.load(path_to_checkpoint, map_location=lambda storage, loc: storage)
        model_mc.load_state_dict(checkpoint['state_dict'], strict=False)
        model_mc = model_mc.eval()
        if torch.cuda.is_available():
            model_mc = model_mc.cuda()
        # ---- load model midc ---- #
        model_nm = VarNetModule(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8)
        path_to_checkpoint = 'D:\DeepLearning_MOCO\training_epoches\epoch=75.ckpt'
        checkpoint = torch.load(path_to_checkpoint, map_location=lambda storage, loc: storage)
        model_nm.load_state_dict(checkpoint['state_dict'], strict=False)
        if torch.cuda.is_available():
            model_nm = model_nm.cuda()
        # --- svm classifier --- #
        classifier = pickle.load(open('./scratch/svm_classifier_model', 'rb'))
        
        # --- read data --- #
        fnames = sorted(glob.glob('D:\DeepLearning_MOCO\training_epoches\multicoil_val\*.h5'))
        data_log = {'filename': ['reference_image', 'motionsim_image', 'motionsim_und_image', 'recon_midc', 'recon_no_midc', 'svm_flag', 'scores_midc', 'scores_no_midc', 'scores_svm']}
        #svm_flag: 0: no motion detected, 1: motion detected
        for k, fname in tqdm(enumerate(fnames)):
            hf = h5py.File(fname, 'r')
            volume_kspace = np.array(hf['kspace'][()], dtype='complex64')
            numSlc, numColis, height, width = volume_kspace.shape
            ### ------- choose one slice amd recon ref image------ ###
            for slc in range(1, 2):
                try:
                    slice_kspace = volume_kspace[slc]
                    slice_kspace_tensor = T.to_tensor(slice_kspace)
                    if torch.cuda.is_available():
                        slice_kspace_tensor = slice_kspace_tensor.cuda()
                    slice_image = fastmri.ifft2c(slice_kspace_tensor)
                    slice_image_abs = fastmri.complex_abs(slice_image)
                    slice_image_rss = np.array(T.center_crop(fastmri.rss(slice_image_abs, dim=0), (H, W)).cpu().detach().numpy())

                    ## ----- Simulate motion ------ #
                    slice_kspace_tensor = torch.squeeze(mocoSimUnd(torch.unsqueeze(slice_kspace_tensor, 0)), 0)
                    slice_image_rss_ms = np.array(T.center_crop(fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(slice_kspace_tensor)), dim=0), (H, W)).cpu().detach().numpy())
                    
                    ### ----- Undersample k-space ------ #
                    masked_kspace, mask = T.apply_mask(slice_kspace_tensor.cpu(), mask_func, seed=0)
                    if torch.cuda.is_available():
                        masked_kspace, mask = masked_kspace.cuda(), mask.cuda()
                    sampled_image = fastmri.ifft2c(masked_kspace)
                    sampled_image_abs = fastmri.complex_abs(sampled_image)
                    sampled_image_rss = np.array(T.center_crop(fastmri.rss(sampled_image_abs, dim=0), (H, W)).cpu().detach().numpy())
                    
                    ## --- predict no-motion-model --- #
                    output_nm = model_nm(torch.unsqueeze(masked_kspace, 0), mask.byte())[-1]
                    output_nm = T.center_crop(np.squeeze(output_nm.cpu().detach().numpy()), (H, W))

                    ### ----- predict midc ----- ###
                    output_mc, dc_weights = model_mc(torch.unsqueeze(masked_kspace, 0), mask.byte())
                    output_mc = T.center_crop(np.squeeze(output_mc[-1].cpu().detach().numpy()), (H, W))
                    dc_weight_np = np.array([float(dc_weight.cpu().detach().numpy()) for dc_weight in dc_weights]).reshape((1, 6))
                    svm_flag = classifier.predict(dc_weight_np)
                    
                    # --- score compute --- #
                    scores_before = compute_scores(slice_image_rss, sampled_image_rss)
                    scores_midc = compute_scores(slice_image_rss, output_mc)
                    scores_no_midc = compute_scores(slice_image_rss, output_nm)
                    scores_svm = scores_midc if svm_flag else scores_no_midc
                    print(fname, '\nscores_before:', scores_before, '\nscores_midc:', scores_midc, '\nscores_no_midc:', scores_no_midc, '\nscores_svm:', scores_svm, '\nsvm_flag: ', svm_flag, '\n')
                    
                    # --- log --- #
                    # {'filename': ['reference_image', 'motionsim_image', 'motionsim_und_image', 'recon_midc', 'recon_no_midc', 'svm_flag', 'scores_midc', 'scores_no_midc', 'scores_svm']}
                    data_log[fname.split('/')[-1]] = [slice_image_rss.astype('float32'), slice_image_rss_ms.astype('float32'), sampled_image_rss.astype('float32'), output_mc.astype('float32'), output_nm.astype('float32'), svm_flag, scores_midc, scores_no_midc, scores_svm]
                except:
                    print('not processing: ', k+1, slc+1)
                                  
        file = open(filenames_list[indx],'wb')
        pickle.dump(data_log, file)
        file.close()







