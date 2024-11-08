#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import sys
import nibabel as nib
import numpy as np
from pathlib import Path
import subprocess
from pathlib import Path
from nilearn.image import mean_img, new_img_like

#exp_dir = Path(f'../../../figures/searchlight/')
current_dir=Path().absolute()
out_dir = Path(f'../../../results/searchlight/')

files = ['PERCEPTION.nii.gz', 'BASELINE.nii.gz','NAIVE.nii.gz', 'RTLC.nii.gz','NAIVEminusBASELINE.nii.gz','RTLCminusNAIVE.nii.gz'
	'BW.nii.gz', 'BWminusNAIVE.nii.gz','RTLCminusBW.nii.gz'
        ]
files = ['BWminusNAIVE.nii.gz','RTLCminusBW.nii.gz','BWminusRTLC.nii.gz']
for filename in files:
    file = out_dir / filename
    
    if 'minus' in filename:
        file = str(file).split("minus")[0] + ".nii.gz"
    if not Path(file).is_file(): print(f'{file} not found'); continue
    outfile = str(out_dir / (filename.split('.nii.gz')[0] + "_projectedaccuracy.nii.gz"))
    if Path(outfile).is_file(): print(f'{outfile} is already there'); continue
    
    fmri_img = nib.load(str(file))
    fmri_data = fmri_img.get_fdata().copy()
    fmri_data = np.average(fmri_data, axis = 3)
    fmri_data = fmri_data + 0.5
    
    
    clusters_file = filename.split(".nii.gz")[0] + "OneSampT_tfce_corrp_tstat1.nii.gz"
    clusters_file = str(out_dir / clusters_file)
    if not Path(clusters_file).is_file(): print(f'{clusters_file} not found'); continue
    print(clusters_file)
    # else:
    #     clusters_file = filename.split("minus")[0] + "OneSampT_tfce_corrp_tstat1.nii.gz"
    #     clusters_file = str(current_dir / clusters_file)
    clusters_img = nib.load(clusters_file)
    clusters_data = clusters_img.get_fdata().copy()
    projected_img = fmri_data.copy() if 'minus' not in filename else clusters_data.copy()
    
    
    if 'minus' in filename:
         clusters_file2 = filename.split("minus")[0] +  "_projectedaccuracy.nii.gz"
         clusters_file2 = str(out_dir / clusters_file2)
         clusters_img2 = nib.load(clusters_file2)
         clusters_data2 = clusters_img2.get_fdata().copy()
         projected_img[np.where(~((clusters_data2>0) & (clusters_data > .95)))] =0

    else:
    	projected_img[np.where(clusters_data <= .95)] = 0
    # print(np.min(projected_img),np.max(projected_img))
    new_nii = new_img_like(clusters_img,projected_img)#, clusters_img.affine, clusters_img.header)
    nib.save(new_nii, outfile)
    
    
    

    
     
