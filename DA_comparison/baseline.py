#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:07:14 2023

@author: alexolza
Usage: 
python baseline.py source_domain:str target_domain:str subject:int region:int(-1 to use all regions) NITER:int with_tgt:int(0 for BASELINE, 1 for NAIVE)
"""

""" THIRD PARTY IMPORTS """
import sys
import os
sys.path.append('..')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
from time import time
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  StratifiedGroupKFold
from tqdm import tqdm
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

""" OWN IMPORTS """

from dataManipulation.loadDataDA import DomainAdaptationData, DomainAdaptationSplitter
from dataManipulation.loadData import MyFullDataset

""" VARIABLE DEFINITION """

subjects = sorted([S.split('/')[-1] for S in glob.glob(os.path.join('../data','perception','*'))])
allregions=sorted([R.split('/')[-1].split('.')[0] for R in glob.glob(os.path.join('../data','perception/1','*.npy'))])

source_domain =  sys.argv[1]
target_domain =  sys.argv[2]
subject = subjects[ int(eval(sys.argv[3]))]
region = allregions if int(eval(sys.argv[4]))==-1 else allregions[int(eval(sys.argv[4]))]
NITER = int(eval(sys.argv[5]))
concat_tgt=int(eval(sys.argv[6]))
concat_tgt_marker='' if concat_tgt==0 else '_withtgt'
splitting='StratifiedGroupKFold'
n_folds = 5
region_name='all_regions' if int(eval(sys.argv[4]))==-1 else region

outdir = os.path.join('../results/DA_comparison', region_name, f'{source_domain}_{target_domain}', subject)
if not os.path.exists(outdir):
    os.makedirs(outdir)

""" MAIN PROGRAM """
#%%
fname = os.path.join(outdir, f'baseline{concat_tgt_marker}.csv')
print(f'Fitting baseline {concat_tgt_marker} for subject {subject} in region {region_name}...')
if not Path(fname).is_file():
    
    t0=time()      

    results={}
    t0=time()
    
    x=range(10,110, 10)
    
    remove_noise=True

    perc_X, perc_y, perc_g = MyFullDataset(source_domain, subject, region, remove_noise=remove_noise)[:]
    imag_X, imag_y, imag_g = MyFullDataset(target_domain, subject, region, remove_noise=remove_noise)[:]

    estimator = LogisticRegression


    balanced_accuracy, balanced_accuracy_im, balanced_accuracy_imtr=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    predictions = pd.DataFrame()
    for shots in tqdm(x):
        balanced_accuracy_s,balanced_accuracy_im_s, balanced_accuracy_imtr_s=[],[],[]
        s = DomainAdaptationSplitter(StratifiedGroupKFold, NITER)
        Source, Target=s.split(perc_X, perc_y, perc_g,imag_X, imag_y, imag_g,shots,shots)# last arg is random seed
        d = DomainAdaptationData(Source, Target)
        for i in range(NITER):
            clf = estimator(n_jobs=-1)
            train = d.Source_train_X[i]
            test = d.Source_test_X[i]

            train_label = np.ravel(d.Source_train_y[i])  
            test_label = np.ravel(d.Source_test_y[i])
            # We select a number "shots" of instances from the target domain (usually imagery)
            I_train, I_test, IL_train, IL_test = d.Target_train_X[i], d.Target_test_X[i], d.Target_train_y[i], d.Target_test_y[i]
            # I_train contains "shots" instances. Those are passed to the ADAPT method
            
            
            if concat_tgt==1:
                train = np.vstack([train,I_train])
                # test = np.vstack([test,I_test])
                
                train_label = np.hstack([train_label,IL_train])
                # test_label = np.vstack([test_label,IL_test])
    
            clf=clf.fit(train,train_label)
            aux_ys = clf.predict(test)                 # Predictions in source domain 
            aux_ys_imag = clf.predict(I_test)       # Predictions in target domain
            aux_ys_imag_tr = clf.predict(I_train)
            

            
            balanced_accuracy_s.append(balanced_accuracy_score( test_label, aux_ys))
            
            balanced_accuracy_im_s.append(balanced_accuracy_score( IL_test, aux_ys_imag))
            balanced_accuracy_imtr_s.append(balanced_accuracy_score( IL_train, aux_ys_imag_tr))
            
        balanced_accuracy[shots]=balanced_accuracy_s      
        balanced_accuracy_im[shots]=balanced_accuracy_im_s
        balanced_accuracy_imtr[shots]=balanced_accuracy_imtr_s

    balanced_accuracy['Domain']=source_domain
    balanced_accuracy_im['Domain']=target_domain
    balanced_accuracy_imtr['Domain']=target_domain+'_tr'
    results= pd.concat([balanced_accuracy,balanced_accuracy_im,balanced_accuracy_imtr])
    results.to_csv(fname)
    print(f'Done {subject}: ',time()-t0)
else:
     print('Nothing done: The result file already exists.')
