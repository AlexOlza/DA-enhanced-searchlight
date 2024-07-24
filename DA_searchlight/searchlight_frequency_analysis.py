#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alexolza

Usage: python searchlight_frequency_analysis.py NITER:int
"""


import os
import re
import sys
from pathlib import Path
from matplotlib import pyplot as plt
sys.path.append('..')
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
source_domain = 'perception'
target_domain = 'imagery'
radius = 12
NITER= sys.argv[1]
savefig_dir = '../figures/searchlight'
if not os.path.exists(savefig_dir):
	os.makedirs(savefig_dir)
out_dir = '../results/searchlight'
data_dir = '../data/whole_brain/perception/'
fname=savefig_dir+'/best_freq.png'

subjects = sorted([S.split('/')[-1] for S in glob(os.path.join(out_dir,'*'))])
print(subjects)
best_freq_df = pd.DataFrame(columns=[['BASELINE','NAIVE','RTLC']])
for i, subject in enumerate(subjects):
	s = re.sub('[0-9]+_','',subject).capitalize()

	pbase = os.path.join(out_dir,subject,f'map_{NITER}iter_{radius}mm_BASELINE.csv')
	pnaive = os.path.join(out_dir,subject,f'map_{NITER}iter_{radius}mm_NAIVE.csv')
	pda = os.path.join(out_dir,subject,f'map_{NITER}iter_{radius}mm_RTLC.csv')
	pperc = os.path.join(out_dir,subject,f'map_{NITER}iter_{radius}mm_PERCEPTION.csv')
	is_file = {'baseline': Path(pbase).is_file(), 'naive': Path(pnaive).is_file(), 'da': Path(pda).is_file(), 'PERC':Path(pperc).is_file()}
	print(is_file)
	searchlight_naive = pd.read_csv(pnaive)
	searchlight = pd.read_csv(pbase)
	searchlight_DA = pd.read_csv(pda)
	searchlight_perc = pd.read_csv(pperc)
	#%%
	searchlight_mean = searchlight.mean(axis=1)
	searchlight_DA_mean =  searchlight_DA.mean(axis=1)
	searchlight_naive_mean = searchlight_naive.mean(axis=1)
	searchlight_perc_mean = searchlight_perc.mean(axis=1)
	df = pd.DataFrame()
	df['PERCEPTION']=searchlight_perc_mean
	df['BASELINE']=searchlight_mean
	df['NAIVE']=searchlight_naive_mean
	df['RTLC']=searchlight_DA_mean
	

	print(s, df.shape[0])
	idxmax = df.drop('PERCEPTION',axis=1).idxmax(axis=1)
	best = np.zeros_like(df.drop('PERCEPTION',axis=1).to_numpy())
	best[:,0] = np.where((idxmax=='BASELINE') ,1,0)
	best[:,1] = np.where((idxmax=='NAIVE') ,1,0)
	best[:,2] = np.where((idxmax=='RTLC') ,1,0)
	best_freq = 100*best.sum(axis=0)/len(best)
	best_freq_df.loc[i+1]=best_freq
	
	
print(best_freq_df)

fig, ax = plt.subplots()
sns.heatmap(best_freq_df,annot=True,ax=ax)
plt.xlabel('Algorithm')
plt.ylabel('Subject')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(fname)
