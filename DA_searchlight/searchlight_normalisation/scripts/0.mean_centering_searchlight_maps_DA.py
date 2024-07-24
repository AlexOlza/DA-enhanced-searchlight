#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import listdir
import nibabel as nib
import numpy as np
from pathlib import Path
import warnings
import sys
warnings.simplefilter("error") # Convert warnings into an error to traceback
algorithm = sys.argv[1]
NITER=sys.argv[2]
if algorithm in ['BASELINE','NAIVE','RTLC', 'PERCEPTION']: offset = 0.5
else: offset = 1
print('Centering offset: ', offset)
# SET FILE STRUCTURE
exp_dir = Path('../../../results/searchlight/')
sys.path.append(exp_dir)

searchlight_maps = sorted(exp_dir.glob(f'*/map_{NITER}iter_12mm_{algorithm}.nii.gz')) 

for result in searchlight_maps:
    
    fmri_img = nib.load(str(result))
    fmri_img.header.set_xyzt_units(2)
    data = fmri_img.get_fdata()
    new_data = data.copy()
    new_data = new_data - offset
    new_nii = nib.Nifti1Image(new_data, fmri_img.affine, fmri_img.header)
    name = result.name.split('.nii.gz')[0]
    nib.save(new_nii, str(result.parent / f'{name}_centered.nii.gz'))
