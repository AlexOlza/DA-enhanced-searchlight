#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import nibabel as nib
import numpy as np
from pathlib import Path
import subprocess
from pathlib import Path
from nilearn.image import new_img_like, load_img
# SET FILE STRUCTURE
exp_dir = Path().absolute()
print(exp_dir)

projected_files = sorted(exp_dir.glob('../../../results/searchlight/*_projectedaccuracy.nii.gz')) # baseline, naive, perception, rtlc, bw, rtlcminusnaive, baselineminusnaive, bwminusnaive, rtlcminusbw
minus = [p for p in projected_files if 'minus' in str(p)]
print(minus)
print(projected_files)
# assert False

#%%
indx = [2,0,1,3] #perception, baseline, naive, rtlc
original_files = np.array([list(exp_dir.glob(f'../../../results/searchlight/{alg}_projectedaccuracy.nii.gz')) for alg in ['PERCEPTION','BASELINE','NAIVE','RTLC']]).ravel()#[p for p in projected_files if (('minus' not in str(p)) and ('BW' not in str(p)))] # only the algorithms in the original manuscript, excluding alg-to-alg comparisons
#projected_files =[projected_files[_ind] for _ind in indx]
projected_files = np.concatenate([original_files , minus])
# PLOTTING
from nilearn import plotting, surface, datasets
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0, "yellow"), (0.5, "yellow"), (0.75, "orange"), (1.0, "red")])
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0, "magenta"), (0.5, "magenta"), (0.51,'mediumvioletred'),
                                                                (0.59,'mediumvioletred'),
                                                                (0.6,"blue"),(0.75,"green"),(0.8,'yellow'),
                                                                (0.85, "orange"), (1.0, "red")])
colors = [
    (0.8, 0.0, 0.9),
    (1.0, 0.0, 0.0),  # Red
    (1.0, 0.5, 0.0),  # Orange
    (1.0, 1.0, 0.0),  # Yellow
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (0.5, 0.0, 1.0)   # Violet
]

# Define the positions of the colors normalized to [0, 1]
n_colors = len(colors)
values = np.linspace(0, 1, n_colors)

# Create the colormap dictionary
cdict = {'red': [], 'green': [], 'blue': []}
for val, color in zip(values, colors):
    r, g, b = color
    cdict['red'].append((val, r, r))
    cdict['green'].append((val, g, g))
    cdict['blue'].append((val, b, b))

# Create the colormap with higher resolution
cmap = matplotlib.colors.LinearSegmentedColormap('RainbowCMap', segmentdata=cdict, N=1000)


# fig, ax = plt.subplots(len(projected_files),1,figsize=(10,8),sharex=True,sharey=True)
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(4, 2, width_ratios=[ 1, 0.03], wspace=0.3, hspace=0.3)

# Create subplots for the glass brain plots
original_axes = [fig.add_subplot(gs[i, 0]) for i in range(4) ]
fig.subplots_adjust(wspace=0, hspace=0.5)
for ax,file in zip(original_axes,original_files):
    print(file)
    parent_folder = file.parent
    filename = file.name.split('.nii.gz')[0]
    file = str(file)
    img = load_img(file).get_fdata()
    output_file = str(parent_folder / (filename + '.png'))
    statmap = plotting.plot_glass_brain(None, 
                                     display_mode = 'lzry',
                                     #cut_coords = [-46, -32, -16, 18, 32, 50],
                                     #cut_coords = ( 24, 35,34),
                                       threshold = 0.5,
                                     # vmin=0.5,
                                      # cmap = cmap,
                                      # alpha=0.9,
                                      plot_abs=False,
                                      # resampling_interpolation='nearest',
                                     annotate = False,
                                     draw_cross = False,
                                     # colorbar = True,
                                      vmax = 1,
                                     #output_file = output_file,
                                     axes =ax)
    try:
        thr = 0.95 if 'minus' in filename else 0.5
        statmap.add_contours(file,filled=True,cmap=cmap,alpha=0.5,vmin=0.5, threshold=thr,vmax=1)
        
    except:
        pass
    # plt.subplots_adjust(right=1)
    # statmap._show_colorbar=True1
    title = f'{filename.split("_")[0]} ' if 'minus' not in filename else f'{filename.split("_")[0].split("minus")[0]} > {filename.split("_")[0].split("minus")[1]} '
   
    ax.set_title(title,loc='left', fontsize=10, y=1.2, pad =-10,weight='bold')
    norm=plt.Normalize(0.5,1)
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    # cax = divider.appencax = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[:,1])
    cbar = plt.colorbar(sm, cax=cax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Add the colorbar to the new axis
    # cbar = plt.colorbar(sm, cax=cax)
plt.show()
fig.savefig('../../../figures/searchlight/projected_accuracies_glass.pdf')

# Create an individual plot for BW which was added during peer review
BWfile = Path('../../../results/searchlight/BW_projectedaccuracy.nii.gz')
fig, ax = plt.subplots(figsize=(10, 8))
fig.subplots_adjust(wspace=0, hspace=0.5)
parent_folder = BWfile.parent
filename = BWfile.name.split('.nii.gz')[0]
file = str(BWfile)
img = load_img(file).get_fdata()
output_file = str(parent_folder / (filename + '.png'))
statmap = plotting.plot_glass_brain(None, 
			     display_mode = 'lzry',
			     #cut_coords = [-46, -32, -16, 18, 32, 50],
			     #cut_coords = ( 24, 35,34),
			       threshold = 0.5,
			     # vmin=0.5,
			      # cmap = cmap,
			      # alpha=0.9,
			      plot_abs=False,
			      # resampling_interpolation='nearest',
			     annotate = False,
			     draw_cross = False,
			     # colorbar = True,
			      vmax = 1,
			     #output_file = output_file,
			     axes =ax)

thr = 0.5
statmap.add_contours(file,filled=True,cmap=cmap,alpha=0.5,vmin=0.5, threshold=thr,vmax=1)

title = f'{filename.split("_")[0]} '

ax.set_title(title,loc='left', fontsize=10, y=1.2, pad =-10,weight='bold')
norm=plt.Normalize(0.5,1)
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
sm.set_array([])
divider = make_axes_locatable(ax)
# cax = divider.appencax = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[:,1])
cbar = plt.colorbar(sm, cax=cax)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
	spine.set_visible(False)
# Add the colorbar to the new axis
# cbar = plt.colorbar(sm, cax=cax)
plt.show()
fig.savefig('../../../figures/searchlight/BW_projected_accuracies_glass.pdf')

# Create a comparison plot for BW and RTLC, with different color scale
BWfile = Path('../../../results/searchlight/BW_projectedaccuracy.nii.gz')
RTLCfile = Path('../../../results/searchlight/RTLC_projectedaccuracy.nii.gz')
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[ 1, 0.03], wspace=0.3, hspace=0.3)

# Create subplots for the glass brain plots
axes = [fig.add_subplot(gs[i, 0]) for i in range(2) ]
fig.subplots_adjust(wspace=0, hspace=0.5)
parent_folder = BWfile.parent
filename = BWfile.name.split('.nii.gz')[0]
file = str(BWfile)
max_acc =max( load_img(BWfile).get_fdata().max(), load_img(RTLCfile).get_fdata().max())
for ax,file in zip(axes,[BWfile,RTLCfile]):
	print(file)
	parent_folder = file.parent
	filename = file.name.split('.nii.gz')[0]
	file = str(file)
	img = load_img(file).get_fdata()
	output_file = str(parent_folder / (filename + '.png'))
	statmap = plotting.plot_glass_brain(None, 
		                     display_mode = 'lzry',
                             #cut_coords = [-46, -32, -16, 18, 32, 50],
	                             #cut_coords = ( 24, 35,34),
	                               threshold = 0.5,
	                             # vmin=0.5,
	                              # cmap = cmap,
	                              # alpha=0.9,
	                              plot_abs=False,
	                              # resampling_interpolation='nearest',
	                             annotate = False,
	                             draw_cross = False,
	                             # colorbar = True,
	                              vmax = max_acc,
	                             #output_file = output_file,
	                             axes =ax)

	thr = 0.5
	statmap.add_contours(file,filled=True,cmap=cmap,alpha=0.5,vmin=0.5, threshold=thr,vmax=max_acc)

	title = f'{filename.split("_")[0]} '

	ax.set_title(title,loc='left', fontsize=10, y=1.2, pad =-10,weight='bold')
	norm=plt.Normalize(0.5,max_acc)
	sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
	sm.set_array([])
	divider = make_axes_locatable(ax)
	# cax = divider.appencax = fig.add_subplot(gs[1])
	cax = fig.add_subplot(gs[:,1])
	cbar = plt.colorbar(sm, cax=cax)
	ax.set_xticks([])
	ax.set_yticks([])
	for spine in ax.spines.values():
		spine.set_visible(False)
	# Add the colorbar to the new axis
	# cbar = plt.colorbar(sm, cax=cax)
plt.show()
fig.savefig('../../../figures/searchlight/BW_RTLC_projected_accuracies_glass.pdf')

# Now create individual plots for each alg-to-alg comparison (those with name ALG1minusALG2). No colorbar here, because we are showing significance clusters instead of projected accuracy.
for file in minus:
	print(file)
	fig, ax = plt.subplots(figsize=(10, 5))
	fig.subplots_adjust(wspace=0, hspace=0.5)
	
	parent_folder = file.parent
	filename = file.name.split('.nii.gz')[0]
	file = str(file)
	img = load_img(file).get_fdata()
	output_file = str(parent_folder / (filename + '.png'))
	statmap = plotting.plot_glass_brain(None, 
			             display_mode = 'lzry',
			             #cut_coords = [-46, -32, -16, 18, 32, 50],
			             #cut_coords = ( 24, 35,34),
			               threshold = 0.5,
			             # vmin=0.5,
			              # cmap = cmap,
			              # alpha=0.9,
			              plot_abs=False,
			              # resampling_interpolation='nearest',
			             annotate = False,
			             draw_cross = False,
			             # colorbar = True,
			              vmax = 1,
			             #output_file = output_file,
			             axes =ax)
	
	thr = 0.95
	statmap.add_contours(file,filled=True,cmap=cmap,alpha=0.5,vmin=0.5, threshold=thr,vmax=1)

	title = f'{filename.split("_")[0]} ' if 'minus' not in filename else f'{filename.split("_")[0].split("minus")[0]} > {filename.split("_")[0].split("minus")[1]}'

	ax.set_title(title,loc='left',y=1,pad=-60, fontsize=10, weight='bold')
	ax.set_xticks([])
	ax.set_yticks([])
	for spine in ax.spines.values():
		spine.set_visible(False)
	# Add the colorbar to the new axis
	# cbar = plt.colorbar(sm, cax=cax)
	plt.show()
	fig.savefig(f'../../../figures/searchlight/{filename}_projected_accuracies_glass.pdf')

