#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 10 2024

@author: aolza
"""
import subprocess
import nibabel as nib
import numpy as np
from nilearn import masking
from pathlib import Path
import glob
from nilearn.image import new_img_like
from tqdm import tqdm
import warnings
import os
import pandas as pd
import sys
sys.path.append('../../..')
from DA_searchlight.transform_results import load_data

warnings.simplefilter("error") # Convert warnings into an error to traceback
algorithm = sys.argv[1]
NITER = sys.argv[2]
fsl_location=sys.argv[3].split('.')[0]
if algorithm.upper() in ['BASELINE','NAIVE','RTLC', 'PERCEPTION']: offset = 0.5
else: offset = 1
print('Centering offset: ', offset)
# SET FILE STRUCTURE
exp_dir = Path('../../../results/searchlight/')

current_dir=Path().absolute()

participant_names = sorted(Path(exp_dir).glob('*'))
subjects = []
radius=12
for result in participant_names:
    print(result)
    participant = result.name
    if result.is_dir(): subjects.append(participant)
    
print(subjects)

for ss in  subjects:
	print(ss)
	_,_,masker = load_data('imagery',ss,main_data_dir='../../../data')
	try:
		process_mask, process_mask_affine = masking.load_mask_img(
			masker.mask_img
		)
	except AttributeError:
		process_mask, process_mask_affine = masking._load_mask_img(
			masker.mask_img
		)
	pda = os.path.join(exp_dir,ss,f'map_{NITER}iter_{radius}mm_{algorithm}.csv')
	df = pd.read_csv(pda)

	outdir = exp_dir/ ss
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	mni_files = ''
	non_mni_files=''
	print('Converting to mni')
	for i,col in tqdm(enumerate(df.columns)):	
		fname = f'{algorithm.upper()}_{NITER}iter_{radius}mm_{i}_centered.nii.gz'
		fname_mni = f'{algorithm.upper()}_{NITER}iter_{radius}mm_{i}_centered_MNI.nii.gz'
		f = os.path.join(outdir,fname)
		non_mni_files+=' '+f
		scores_3D = np.zeros(process_mask.shape)
		scores_3D[process_mask] = df[col]-offset
		searchlight_img_0 = new_img_like(masker.mask_img, scores_3D)
		searchlight_img_0.to_filename(f)

		participant_dir = current_dir.parent / f'participants/{ss}'
		
		anat_dir = participant_dir / 'perception/raw/anat'
		mni_example_func_dir = participant_dir / 'perception/preprocessed/mni/example_func'
		nonlinear_trans_nii = anat_dir / 'nonlinear_trans.nii.gz'
		output_mat_file = mni_example_func_dir / 'examplefunc2struct.mat'
		searchlight_to_mni_file = os.path.join(outdir,fname_mni)
		mni_files+=' '+searchlight_to_mni_file
		subprocess.run([f"applywarp --ref={fsl_location} --in={f} --warp={nonlinear_trans_nii} --premat={output_mat_file} --out={searchlight_to_mni_file}"], 			shell = True)
	print('Merging')
	subprocess.run([f"fslmerge -t {outdir}/{algorithm} {mni_files}"], shell = True)
	print('Randomise')
	subprocess.run([f"randomise -i {outdir}/{algorithm} -o {outdir}/{algorithm.upper()}OneSampT -1 -v 6 -T -m /home/alexolza/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz -n 5000"], shell = True)
