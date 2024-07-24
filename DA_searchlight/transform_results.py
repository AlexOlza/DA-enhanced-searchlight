#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:33:02 2024

@author: alexolza
"""
import tensorflow as tf
import os
import re
import time
import sys
sys.path.append('..')
from sklearn.model_selection import cross_val_score
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import nilearn
from nilearn import plotting
from nibabel import load as load_fmri
from nilearn import masking
from nilearn.maskers import NiftiMasker
from nilearn.image.resampling import coord_transform
from nilearn.image import index_img, new_img_like
from nilearn.decoding.searchlight import search_light, GroupIterator,_group_iter_search_light
import warnings
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, cpu_count, delayed
from sklearn.linear_model import LogisticRegression
from dataManipulation.loadDataDA import DomainAdaptationSplitter
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity
from matplotlib import pyplot as plt
##

#%%

#%%
def load_data(domain, subject,NITER=100,main_data_dir='../data'):
    
    
    data_dir =f'{main_data_dir}/whole_brain/{domain}/{subject}' 
    event_dir = f'{main_data_dir}/{domain}/{subject}/events.csv'
    
    
    events = pd.read_csv(event_dir, usecols=['trial_idx','run','target_category'])
    white_noise_events_idx = np.where(events.target_category==2)[0]
    events = events.drop(white_noise_events_idx)
    
    
    bold_data = load_fmri(os.path.join(data_dir,f'{domain}_detrended_zscored_stacked_vols_of_interest_searchlight.nii.gz'))

    # I discard the scans corresponding to white noise
    bold_data = index_img(bold_data,events.index)
    
    example_func = os.path.join(re.sub(domain,'perception',data_dir),'example_func_deoblique_brainmask.nii')

    ex_f = load_fmri(example_func)
    masker = NiftiMasker(ex_f).fit()
    
    return(bold_data, events, masker)

