#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path
import sys
# SET FILE STRUCTURE
algorithm = sys.argv[1]
fsl_location=sys.argv[2]
# SET FILE STRUCTURE
exp_dir = Path('../../../results/searchlight/')
sys.path.append(exp_dir)
print(exp_dir)
current_dir = Path().absolute()

searchlight_maps = sorted(exp_dir.glob(f'*/*12mm_{algorithm}.nii.gz'))

for result in searchlight_maps:
    
    participant = result.parent.name
    print (participant)
    participant_dir = current_dir.parent / f'participants/{participant}'
   
    anat_dir = participant_dir / 'perception/raw/anat'
    
    mni_example_func_dir = participant_dir / 'perception/preprocessed/mni/example_func'
    
    nonlinear_trans_nii = anat_dir / 'nonlinear_trans.nii.gz'
    output_mat_file = mni_example_func_dir / 'examplefunc2struct.mat'
    
    name = result.name.split('.nii.gz')[0]
    searchlight_to_mni_file = result.parent / f'{name}_nocentered_mni.nii.gz'
    subprocess.run([f"applywarp --ref={fsl_location} --in={result} --warp={nonlinear_trans_nii} --premat={output_mat_file} --out={searchlight_to_mni_file}"], shell = True)
    
