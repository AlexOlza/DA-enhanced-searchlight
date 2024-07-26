#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:17:01 2021

@author: pmargolles
"""
import sys
import nibabel as nib
import numpy as np
from pathlib import Path
from pathlib import Path
from nilearn.image import mean_img
exp_dir = Path('../../../results/searchlight/')
current_dir=Path().absolute()
sys.path.append(exp_dir)
grouped_searchlights_dir = exp_dir 
files = ['PERCEPTION.nii.gz', 'BASELINE.nii.gz','NAIVE.nii.gz', 'RTLC.nii.gz'
        ]

for filename in files:
    file = grouped_searchlights_dir / filename
    print(file)
        
    fmri_img = nib.load(str(file))
    fmri_data = fmri_img.get_fdata().copy()
    fmri_data = np.average(fmri_data, axis = 3)
    fmri_data = fmri_data + 0.5
    
    clusters_file = f"{str(file).split('.nii.gz')[0]}OneSampT_tfce_corrp_tstat1.nii.gz"
    clusters_img = nib.load(clusters_file)
    clusters_data = clusters_img.get_fdata().copy()
    projected_img = fmri_data.copy()
    projected_img[np.where(clusters_data <= .95)] = 0
    print(np.min(projected_img),np.max(projected_img), np.unique(projected_img))
    new_nii = nib.Nifti1Image(projected_img, clusters_img.affine, clusters_img.header)
    new = new_nii.get_fdata()
    print(np.min(new),np.max(new), np.unique(new))
    filename = filename.split('.nii.gz')[0]
    nib.save(new_nii, str(exp_dir / (filename + "_projectedaccuracy.nii.gz")))
    
    
    
     
