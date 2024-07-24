#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 9 2024

@author: aolza
"""
import sys
import pandas as pd
import seaborn as sns
import nibabel as nib
import numpy as np
from pathlib import Path
from pathlib import Path
from nilearn.image import index_img, new_img_like
# PLOTTING
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.colors
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0, "gray"), (0.5, "yellow"), (0.75, "yellow"), (1.0, "red")])

# SET FILE STRUCTURE
exp_dir = Path('../../../results/searchlight')
fig_dir = Path('../../../figures/searchlight')
sys.path.append(exp_dir)
current_dir=Path().absolute()

searchlight_maps = sorted(exp_dir.glob(f'*/RTLC.nii.gz'))
subjects = []
for result in searchlight_maps:
    print(result)
    participant = result.parent.name
    subjects.append(participant)

print(subjects)

grouped_searchlights_dir = fig_dir 
algorithm = sys.argv[1]


filename = f'{algorithm.upper()}.nii.gz'
file = exp_dir / filename
print(file)
fmri_img = nib.load(str(file))
n_maps = fmri_img.shape[-1]

fig, ax = plt.subplots(int(np.ceil(n_maps/2)),2,sharex=True,sharey=True, figsize=(20,20))
    
fig.subplots_adjust(wspace=0.2, hspace=0.2)

for i, s in zip( range(n_maps), subjects):
    print(s)
    clusters_file = f"{s}/{algorithm.upper()}OneSampT_tfce_corrp_tstat1.nii.gz"
    clusters_file = str(exp_dir / clusters_file)
    clusters_img = nib.load(clusters_file)
    clusters_data = clusters_img.get_fdata().copy()
    fmri_img_i = index_img(fmri_img,i)
    
    fmri_data = fmri_img_i.get_fdata().copy()  + 0.5
    
    
    projected_img = fmri_data.copy()
    projected_img[np.where(clusters_data <= .95)] = 0
    #print(np.min(projected_img),np.max(projected_img))
    new_nii = nib.Nifti1Image(projected_img, fmri_img_i.affine, fmri_img_i.header)
	
    #new = new_nii.get_fdata()
    #print(np.min(new),np.max(new), np.unique(new))
    #filename = filename.split('.nii.gz')[0]
    #nib.save(new_nii, str(current_dir / (filename + f"{s}_projectedaccuracy.nii.gz")))
    median = np.median(projected_img[projected_img>0]) if len(projected_img[projected_img>0])>0 else 0
    maxim = np.max(projected_img[projected_img>0]) if len(projected_img[projected_img>0])>0 else 0
    niimg = new_nii.get_fdata().copy()
    niimg[niimg>0.5] = (niimg[niimg>0.5]-0.5)/(maxim-0.5)
  
    t=0.5
    statmap = plotting.plot_stat_map(new_nii, 
		             display_mode = 'x',
		             cut_coords = [ -32, -16, 18, 32],
		             #cut_coords = ( 24, 35,34),
		             threshold = t,
		             cmap = cmap,#'YlOrRd',#
		             annotate = True,
		             draw_cross = False,
		             colorbar = True,
		             vmax = 1,
		             #output_file = output_file,
		             axes =ax.flatten()[i])
                            
    ax.flatten()[i].set_title(f'{filename.split(".")[0]} - Subject {i+1}: median={median:.02f}, max={maxim:.02f}',loc='left', fontsize=10, y=1, pad =-10,weight='bold')

fig.savefig(f'{fig_dir}/subj_variability_{filename.split(".")[0]}.pdf',bbox_inches='tight')



