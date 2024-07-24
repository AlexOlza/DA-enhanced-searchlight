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
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

random.seed(0)
np.random.seed(0)

DATAPATH = '../data'
    
class MyFullDataset():
    def __init__(self, domain, subject, regions, remove_noise=True, train=True, all_trials=False, 
                 data_dir=DATAPATH):
        if isinstance(regions, str):
            regions = [regions]
        self.regions=regions 
        self.file = [f'{data_dir}/{domain}/{subject}/{region}.npy' for region in self.regions]
        self.events = pd.DataFrame()
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
    def __getitem__(self, index):

        y = self.labels[index]
        trials = self.trials[index]
        x = self.datax[index]            
        return x, y, trials
    
    def __len__(self):
        return len(self.datax)

