#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:33:02 2024

@author: alexolza
python perception_searchlight.py subject:int NITER:int 
"""

import os
import re
import sys
import warnings
import glob
sys.path.append('..')

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

import pandas as pd
from tqdm import tqdm
import numpy as np
from nilearn import masking
from nilearn.image import new_img_like
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from algorithms.searchlight import get_sphere_data, searchlight_cv_DA, search_light
from dataManipulation.whole_brain import load_data
from pathlib import Path
#%%
source_domain = 'perception'
target_domain = 'perception'
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


#%%
searchlight= pd.DataFrame()

pbase = os.path.join(out_dir,f'map_{NITER}iter_{radius}mm_PERCEPTION.csv')

if not Path(pbase).is_file():
    bold_data, events, masker = load_data(source_domain, subject)
    tgt_data, tgt_events, tgt_masker = load_data(target_domain, subject)


    #%%
    g = events.run+'trial'+events.trial_idx.astype(str)
    g_tgt = tgt_events.run+'trial'+tgt_events.trial_idx.astype(str)


    X,A,_,_ = get_sphere_data(masker,bold_data,radius,)
    X_tgt,A_tgt,_,_ = get_sphere_data(tgt_masker,tgt_data,radius,)


    Source_train_is,Target_train_is = searchlight_cv_DA(events.target_category.values,tgt_events.target_category.values, g,g_tgt,NITER )

    for i in tqdm(range(NITER)):
    
        xtrain = X[Source_train_is[i]]
        ytrain = events.target_category.values[Source_train_is[i]]
        
        xtest = np.delete(X,Source_train_is[i],axis=0)
        ytest = np.delete(events.target_category.values,Source_train_is[i],axis=0)
        searchlight[i] = search_light(xtrain, ytrain,LogisticRegression,A,xtest,ytest,scoring = balanced_accuracy_score, verbose=0) 
        
    searchlight.to_csv(pbase,index=False) 
    
    #%%
    searchlight_mean = searchlight.mean(axis=1)
    
    #%%
    
    # The following try/except depends on the version of nilearn
    try:
     	process_mask, process_mask_affine = masking.load_mask_img(
     	    tgt_masker.mask_img
     	)
    except: 
     	process_mask, process_mask_affine = masking._load_mask_img(
        tgt_masker.mask_img
    )
    scores_3D = np.zeros(process_mask.shape)
    scores_3D[process_mask] = searchlight_mean.values
    
    
    searchlight_img_0 = new_img_like(tgt_masker.mask_img, scores_3D)
    
    # save nifti images
    
    searchlight_img_0.to_filename(os.path.join(out_dir,f'map_{NITER}iter_{radius}mm_PERCEPTION.nii.gz'))
else:
    print('Results file found, nothing done.')

