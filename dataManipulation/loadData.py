#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:27:54 2023

@author: alexolza
"""
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import re
import os
import glob
import random
import bdpy
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

random.seed(0)
np.random.seed(0)

DATAPATH = '../data'
DATAPATH_ds001246 = '../ds001246'
class MyFullDataset(): # Loads the whole dataset for a domain, subject and region
    def __init__(self, domain, subject, regions, remove_noise=True, data_dir=DATAPATH,dataset='own',average=False):
        if dataset == 'own':            
            if isinstance(regions, str):
                regions = [regions]
            self.regions=regions 
            self.file = [f'{data_dir}/{domain}/{subject}/{region}.npy' for region in self.regions]

            x = pd.DataFrame()
            self.events = pd.read_csv(re.sub(f'{regions[0]}.npy','events.csv', self.file[0]),
                                                usecols=['trial_idx','target_category', 'run'])
            self.labels = self.events.target_category
    
            for region,file in zip(self.regions, self.file):
                x = pd.concat([x,pd.DataFrame(np.load(file))],axis=1)
            
            x['trial'] = ((self.events.trial_idx).astype(str)+self.events.run).to_numpy()
            x['y'] = self.labels
            if remove_noise:
                x = x.loc[~ (x.y==2)] # remove the examples where the target category was white noise  
            x = x.reset_index(drop=True)
            self.labels = x.y.to_numpy()
            self.trials = x.trial
            self.datax= x.drop(['trial', 'y'], axis=1).to_numpy()
        
        
        elif dataset=='ds001246_unprocessed':
            
            data_dir = DATAPATH_ds001246 
            event_type='stimulus' if domain=='perception' else 'imagery'
            if isinstance(regions, str):
                regions = [regions]
            averaged_suffix = '_averaged' if average else ''
            self.regions=regions 
            self.file = [f'{data_dir}/{subject}/{domain}_stacked/BOLD{averaged_suffix}/{region}_BOLD.npy' for region in self.regions]
            x = pd.DataFrame()
            events = pd.read_csv( re.sub(f'{regions[0]}_BOLD.npy','events.csv', self.file[0]),
                                                usecols=['trials','targets', 'session'])
            self.events = events
            self.events = self.events.rename({'trials':'trial_idx','targets':'target_category','session':'run'},axis=1)
            self.labels = self.events.target_category
    
            for region,file in zip(self.regions, self.file):
                aux =pd.DataFrame(np.load(file))
                x = pd.concat([x,aux],axis=1)
            x = x.reset_index(drop=True)
            print(len(x),len(self.events),len(events))
            x['trial'] = (self.events.run).to_numpy() if average else  ((self.events.trial_idx).astype(str)+self.events.run).to_numpy()
            x['y'] = self.labels
            
            
            self.labels = x.y.to_numpy()
            self.trials = x.trial
            self.datax= x.drop(['trial', 'y'], axis=1).to_numpy()
        else:
            assert dataset == 'ds001246', 'Accepted values for arg. dataset are "own" (default) or "ds001246" or "ds001246_unprocessed"'
            data_dir = DATAPATH_ds001246 
            if isinstance(regions, str):
                regions = [regions]
            self.regions=regions 
            self.file = f'{data_dir}/{subject}.h5'
            domain_idx = 2 if domain=='perception' else 3
            data = bdpy.BData(self.file)
            region_voxels = data.select(' + '.join([f'ROI_{r}' for r in self.regions]))
            categories = data.select('category_index')
            runs = data.select('Run')
            datatype = data.select('DataType')
            idx = (datatype == domain_idx).flatten()  
            

            self.datax = region_voxels[idx,:]
            self.labels = categories[idx].ravel()
            self.trials = runs[idx].ravel()
            
            
            
    def __getitem__(self, index):

        y = self.labels[index]
        trials = self.trials[index]
        x = self.datax[index]            
        return x, y, trials
    
    def __len__(self):
        return len(self.datax)
