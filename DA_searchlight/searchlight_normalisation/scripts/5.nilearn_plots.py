#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:17:01 2021

@author: pmargolles
"""

from pathlib import Path
from pathlib import Path
# SET FILE STRUCTURE
exp_dir = Path('../../../results/searchlight/')
fig_dir = Path('../../../figures/searchlight/')
print(exp_dir)

projected_files = sorted(exp_dir.glob('**/*_projectedaccuracy.nii.gz')) # baseline, naive, perception, rtlc
indx = [2,0,1,3] #perception, baseline, naive, rtlc
projected_files =[projected_files[_ind] for _ind in indx]

# PLOTTING
from nilearn import plotting, surface, datasets
import matplotlib.pyplot as plt
import matplotlib.colors

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0, "gray"), (0.5, "yellow"), (0.75, "yellow"), (1.0, "red")])
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B

fig, ax = plt.subplots(len(projected_files),1,sharex=True,sharey=True)
fig.subplots_adjust(wspace=0, hspace=0.5)
for i, file in enumerate(projected_files):
    parent_folder = file.parent
    filename = file.name.split('.nii.gz')[0]
    file = str(file)

    output_file = str(parent_folder / (filename + '.png'))
    statmap = plotting.plot_stat_map(file, 
                                     display_mode = 'x',
                                     cut_coords = [-46, -32, -16, 18, 32, 50],
                                     #cut_coords = ( 24, 35,34),
                                     threshold = 0.5,
                                     cmap = cmap,
                                     annotate = True,
                                     draw_cross = False,
                                     colorbar = True,
                                     vmax = 1,
                                     #output_file = output_file,
                                     axes =ax[i])
    ax[i].set_title(f'{filename.split("_")[0]}',loc='left', fontsize=10, y=1, pad =-10,weight='bold')
fig.savefig(str(fig_dir)+'projected_accuracies.pdf')

