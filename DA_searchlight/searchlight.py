#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:33:02 2024

@author: alexolza
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import sys
import glob
import warnings

sys.path.append('..')

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

import pandas as pd
from tqdm import tqdm
import numpy as np
from nilearn import plotting
from nilearn import masking
from nilearn.image import new_img_like
import warnings


from sklearn.linear_model import LogisticRegression
from adapt.parameter_based import RegularTransferLC as RTLC
from adapt.parameter_based import RegularTransferLR
from matplotlib import pyplot as plt
##

from algorithms.searchlight import get_sphere_data, searchlight_cv_DA, search_light
from dataManipulation.whole_brain import load_data
#%%
source_domain = 'perception'
target_domain = 'imagery'
radius = 12
subjects = sorted([S.split('/')[-1] for S in glob.glob(os.path.join('../data','perception','*'))])
subject = subjects[ int(eval(sys.argv[1]))]
s = re.sub('[0-9]+_','',subject).capitalize()
NITER= int(eval(sys.argv[2]))
savefig_dir = f'../figures/searchlight/{subject}'
if not os.path.exists(savefig_dir):
    os.makedirs(savefig_dir)
out_dir = f'../results/searchlight/{subject}'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
from adapt.utils import check_arrays
                         
from sklearn.preprocessing import LabelBinarizer
class RTLC(RegularTransferLR):
	def fit(self, Xt=None, yt=None, **fit_params):       
		Xt, yt = self._get_target_data(Xt, yt)
		Xt, yt = check_arrays(Xt, yt)
		
		_label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
		_label_binarizer.fit(self.estimator.classes_)
		yt = _label_binarizer.transform(yt)
		
		# print(yt.shape) -> this print is present in the current release of ADAPT, and it is very annoying
		
		return super().fit(Xt, yt, **fit_params)

#%%


#%%
searchlight, searchlight_naive, searchlight_DA = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
from pathlib import Path
pbase = os.path.join(out_dir,f'map_{NITER}iter_{radius}mm_BASELINE.csv')
pnaive = os.path.join(out_dir,f'map_{NITER}iter_{radius}mm_NAIVE.csv')
pda = os.path.join(out_dir,f'map_{NITER}iter_{radius}mm_RTLC.csv')
is_file = {'baseline': Path(pbase).is_file(), 'naive': Path(pnaive).is_file(), 'da': Path(pda).is_file()}
if all(is_file.values()):
    print('All results files found, nothing done.')
else:
    
    bold_data, events, masker = load_data(source_domain, subject)
    tgt_data, tgt_events, tgt_masker = load_data(target_domain, subject)
    
    
    #%%
    
    g = events.run+'trial'+events.trial_idx.astype(str)
    g_tgt = tgt_events.run+'trial'+tgt_events.trial_idx.astype(str)
    
    
    X,A,_,_ = get_sphere_data(masker,bold_data,radius,)
    X_tgt,A_tgt,_,_ = get_sphere_data(tgt_masker,tgt_data,radius,)
    
    
    Source_train_is,Target_train_is = searchlight_cv_DA(events.target_category.values,tgt_events.target_category.values, g,g_tgt,NITER )
    
    from sklearn.metrics import balanced_accuracy_score
    # def scoring_fn(estimator, X_test, y_test):
    try:
        process_mask, process_mask_affine = masking.load_mask_img(
            tgt_masker.mask_img
        )
    except: 
        process_mask, process_mask_affine = masking._load_mask_img(
        tgt_masker.mask_img
    )
    for i in tqdm(range(NITER)):
        Target_test_is = np.ones(tgt_events.target_category.values.size, dtype=bool)
        Target_test_is[Target_train_is[i]] = False
        xtrain = X[Source_train_is[i]]
        ytrain = events.target_category.values[Source_train_is[i]]
        xtrain_naive =pd.concat([ pd.DataFrame(xtrain),pd.DataFrame(X[Target_train_is[i]])]).to_numpy()
        ytrain_naive = pd.concat([pd.Series(events.target_category.values[Source_train_is[i]]),
                                           pd.Series(events.target_category.values[Target_train_is[i]])]).to_numpy()
        xtest = X_tgt[Target_test_is]
        ytest = tgt_events.target_category.values[Target_test_is]
        xtrain_tgt = X_tgt[Target_train_is[i]]
        ytrain_tgt = tgt_events.target_category.values[Target_train_is[i]]
        if not is_file['baseline']: searchlight[i] = search_light(xtrain, ytrain,LogisticRegression,A,xtest,ytest,scoring = balanced_accuracy_score, verbose=0) 
        if not is_file['naive']: searchlight_naive[i] = search_light(xtrain_naive, ytrain_naive,LogisticRegression,A,xtest,ytest,scoring = balanced_accuracy_score, verbose=0) 
        if not is_file['da']: searchlight_DA[i] = search_light(xtrain, ytrain,LogisticRegression,A,xtest,ytest,scoring = balanced_accuracy_score,DA = RTLC, X_tgt = xtrain_tgt, y_tgt = ytrain_tgt, verbose=0) 
        
    searchlight.to_csv(pbase,index=False) 
    
    searchlight_naive.to_csv(pnaive,index=False)
    
    searchlight_DA.to_csv(pda,index=False)
    
    #%%
    searchlight_mean = searchlight.mean(axis=1)
    searchlight_DA_mean =  searchlight_DA.mean(axis=1)
    searchlight_naive_mean = searchlight_naive.mean(axis=1)
    #%%
    scores_3D = {}
    i=0
    # searchlight_mean[searchlight_mean<0.5]=1e-5
    # searchlight_DA_mean[searchlight_DA_mean<0.5]=1e-5
    # searchlight_naive_mean[searchlight_naive_mean<0.5]=1e-5
    for mean in (searchlight_mean,searchlight_DA_mean,searchlight_DA_mean/searchlight_mean, searchlight_mean/searchlight_DA_mean):
        scores_3D[i] = np.zeros(process_mask.shape)
        scores_3D[i][process_mask] = mean.values
        i+=1
    
    searchlight_img_0 = new_img_like(tgt_masker.mask_img, scores_3D[0])
    searchlight_img_1 = new_img_like(tgt_masker.mask_img, scores_3D[1])
    searchlight_img_2 = new_img_like(tgt_masker.mask_img, scores_3D[2])
    searchlight_img_3 = new_img_like(tgt_masker.mask_img, scores_3D[3])
    
    # save nifti images
    
    searchlight_img_0.to_filename(os.path.join(out_dir,f'map_{NITER}iter_{radius}mm_BASELINE.nii.gz'))
    searchlight_img_1.to_filename(os.path.join(out_dir,f'map_{NITER}iter_{radius}mm_RTLC.nii.gz'))
    searchlight_img_2.to_filename(os.path.join(out_dir,f'map_{NITER}iter_{radius}mm_RTLCBASE.nii.gz'))
    searchlight_img_3.to_filename(os.path.join(out_dir,f'map_{NITER}iter_{radius}mm_BASERTLC.nii.gz'))
    
    #%%
    fig, ax = plt.subplots(4,1,figsize=(12,16))
    
    plotting.plot_stat_map(searchlight_img_0,bg_img = masker.mask_img,cut_coords=(0,-5,-4),threshold=0.5, title = f'Baseline > 0.5- {s}',axes =ax[0])
    plotting.plot_stat_map(searchlight_img_1,bg_img = masker.mask_img,cut_coords=(0,-5,-4),threshold=0.5, title = f'RTLC  > 0.5- {s}',axes =ax[1])
    plotting.plot_stat_map(searchlight_img_2,bg_img = masker.mask_img,cut_coords=(0,-5,-4),threshold=1, title = f'RTLC/Baseline > 1 - {s}',axes =ax[2])
    plotting.plot_stat_map(searchlight_img_3,bg_img = masker.mask_img,cut_coords=(0,-5,-4),threshold=1, title = f'Baseline/RTLC > 1 - {s}',axes =ax[3])
    fig.savefig(os.path.join(savefig_dir,f'fig_{NITER}iter_{radius}mm.png'))
    
    #%%
    
    scores_3D = {}
    i=0
    for mean in (searchlight_naive_mean,searchlight_DA_mean,searchlight_DA_mean/searchlight_naive_mean, searchlight_naive_mean/searchlight_DA_mean):
        scores_3D[i] = np.zeros(process_mask.shape)
        scores_3D[i][process_mask] = mean
        i+=1
    
    searchlight_img_0 = new_img_like(tgt_masker.mask_img, scores_3D[0])
    searchlight_img_1 = new_img_like(tgt_masker.mask_img, scores_3D[1])
    searchlight_img_2 = new_img_like(tgt_masker.mask_img, scores_3D[2])
    searchlight_img_3 = new_img_like(tgt_masker.mask_img, scores_3D[3])
    # save nifti images
    
    searchlight_img_0.to_filename(os.path.join(out_dir,f'map_{NITER}iter_12mm_NAIVE.nii.gz'))
    
    #%%
    fig, ax = plt.subplots(4,1,figsize=(12,16))
    
    plotting.plot_stat_map(searchlight_img_0,bg_img = masker.mask_img,cut_coords=(0,-5,-4),threshold=0.5, title = f'Naive > 0.5- {s}',axes =ax[0])
    plotting.plot_stat_map(searchlight_img_1,bg_img = masker.mask_img,cut_coords=(0,-5,-4),threshold=0.5, title = f'RTLC  > 0.5- {s}',axes =ax[1])
    plotting.plot_stat_map(searchlight_img_2,bg_img = masker.mask_img,cut_coords=(0,-5,-4),threshold=1, title = f'RTLC/Naive > 1 - {s}',axes =ax[2])
    plotting.plot_stat_map(searchlight_img_3,bg_img = masker.mask_img,cut_coords=(0,-5,-4),threshold=1, title = f'Naive/RTLC > 1 - {s}',axes =ax[3])
    fig.savefig(os.path.join(savefig_dir,f'fig_naive_{NITER}iter_{radius}mm.png'))
