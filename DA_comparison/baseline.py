#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:07:14 2023

@author: alexolza

This script fits a Logistic Regression with data from a particular ROI and subject.
If with_tgt==0, the estimator is fitted on the source domain.
If with_tgt==1, the estimator is fitted on union of the source domain and Nt instances of the target domain, where Nt=10, 20, 30, ..., 100.
Either way, the estimator is evaluated both on the source domain, on the training instances of the target domain (where applicable) 
and in the validation instances of the target domain.

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
import re
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,ConfusionMatrixDisplay 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

""" OWN IMPORTS """

from dataManipulation.loadDataDA import DomainAdaptationData, DomainAdaptationSplitter, DomainAdaptationGOD
from dataManipulation.loadData import MyFullDataset

""" VARIABLE DEFINITION """
source_domain =  sys.argv[1]
target_domain =  sys.argv[2]
dataset = int(eval(sys.argv[7]))
shuffle = False
average=False
binary=False
if dataset==0:
    dataset='own'
    subjects = sorted([S.split('/')[-1] for S in glob.glob(os.path.join('../data','perception','*'))])
    allregions=sorted([R.split('/')[-1].split('.')[0] for R in glob.glob(os.path.join(f'../data/perception/{subjects[0]}','*.npy'))])
    idx = [int(r) for r in sys.argv[4].split('+')]
    region = allregions if int(eval(sys.argv[4]))==-1 else [allregions[i] for i in idx]
    
    region_name='all_regions' if int(eval(sys.argv[4]))==-1 else '-'.join(region)
else:
    dataset ='ds001246'
    subjects = sorted([S.split('/')[-1].split('.')[0] for S in glob.glob(os.path.join(f'../{dataset}','*'))])
    region = sys.argv[4]
    region_name = region
subject = subjects[ int(eval(sys.argv[3]))]


NITER = int(eval(sys.argv[5]))
concat_tgt=int(eval(sys.argv[6]))
concat_tgt_marker='' if concat_tgt==0 else '_withtgt'
splitting='StratifiedGroupKFold'
n_folds = 5

if dataset =='own':
    outdir = os.path.join('../results/DA_comparison', region_name, f'{source_domain}_{target_domain}', subject)
else:
    outdir = os.path.join(f'../results/DA_comparison/{dataset}', region_name, f'{source_domain}_{target_domain}', subject)
if not os.path.exists(outdir):
    os.makedirs(outdir)

""" MAIN PROGRAM """
#%%
fname = os.path.join(outdir, f'baseline{concat_tgt_marker}.csv')
params = {} if dataset=='own' else {'class_weight':'balanced'}
print(f'Fitting baseline {concat_tgt_marker} for subject {subject} in region {region_name} using dataset {dataset}...')
# print(allregions)
if not False:#Path(fname).is_file():
    
    t0=time()      

    results={}
    t0=time()
    
    Nts=range(10,110, 10) if dataset=='own' else [200, 250, 300, 350, 400] # Number of training instances to be used during training when concat_tgt==1.

    # The dataset used for the paper contained additional samples collected while the subject was
    # presented with gaussian noise images. We have not used them in our experiments, hence the variable remove_noise.
    remove_noise=True 
    # We read the complete datasets.
    Source_X, Source_y, Source_g = MyFullDataset(source_domain, subject, region, remove_noise=remove_noise, dataset = dataset,average=average)[:] # Returns: voxels, labels, groups (run + trial)
    Target_X, Target_y, Target_g = MyFullDataset(target_domain, subject, region, remove_noise=remove_noise, dataset = dataset,average=average)[:]
    print('N voxels: ', Source_X.shape[-1])
    # if dataset== 'ds001246': NITER = len(np.unique(Source_g))
    
    
    estimator = LogisticRegression


    balanced_accuracy, balanced_accuracy_im, balanced_accuracy_imtr=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    predictions = pd.DataFrame()
    # Even when concat_tgt==0, we use this for loop to ensure that the validation instances are exactly the same as for the rest of the methods
    # that do receive target somain instances for training. This guarantees that performance results can be analysed as paired observations.
    for Nt in tqdm(Nts):
        balanced_accuracy_Nt,balanced_accuracy_im_Nt, balanced_accuracy_imtr_Nt=[],[],[]
        # This object will ensure independent train/test splits for our experimental design, avoiding data leackage.
        # Recall that, as explained in the paper, samples from the same trial and run are not independent.  
        if dataset=='own':
            s = DomainAdaptationSplitter(StratifiedGroupKFold, NITER) 
            Source, Target=s.split(Source_X, Source_y, Source_g,Target_X, Target_y, Target_g,Nt, Nt)# Last argument is the random seed.
        else:
            Source, Target = DomainAdaptationGOD(Source_X, Target_X, Source_y, Target_y, Source_g, Target_g,target_n=Nt).split()
        d = DomainAdaptationData(Source, Target) #Just a wrapping class for convenience.
        for i in tqdm(range(NITER)):
            clf = estimator(**params)
            train = d.Source_train_X[i]
            test = d.Source_test_X[i]
            I_train, I_test, IL_train, IL_test = d.Target_train_X[i], d.Target_test_X[i], d.Target_train_y[i], d.Target_test_y[i]
            # print(train.shape[0], test.shape[0], I_train.shape[0], I_test.shape[0])
            # break
            if dataset== 'ds001246':
                scaler_source=StandardScaler().fit(train)
                train =scaler_source.transform(train)
                test = scaler_source.transform(test)
                scaler_tgt=StandardScaler().fit(I_train)
                I_train =scaler_tgt.transform(I_train)
                I_test =scaler_tgt.transform(I_test)
                
            train_label = np.ravel(d.Source_train_y[i])  
            test_label = np.ravel(d.Source_test_y[i])
            # print(d.Source_train_i[i][:10])
            if shuffle: np.random.shuffle(train_label); print('SHUFFLING LABELS!!!!!!!')

            # We select a number "Nt" of instances from the target domain (usually imagery)
            # I_train contains "Nt" instances. Those are used for training only if concat_tgt==1.
            # I_test contains those remaining target domain instances that are independent from I_train.
            
            if concat_tgt==1:
                train = np.vstack([train,I_train])
                train_label = np.hstack([train_label,IL_train])

            clf=clf.fit(train,train_label)
            aux_ys = clf.predict(test)                 # Predictions in source domain 
            aux_ys_imag = clf.predict(I_test)          # Predictions in target domain
            aux_ys_imag_tr = clf.predict(I_train)      # Predictions in target domain training instances
            

            
            balanced_accuracy_Nt.append(balanced_accuracy_score( test_label, aux_ys))
            
            balanced_accuracy_im_Nt.append(balanced_accuracy_score( IL_test, aux_ys_imag))
            balanced_accuracy_imtr_Nt.append(balanced_accuracy_score( IL_train, aux_ys_imag_tr))
            
        balanced_accuracy[Nt]=balanced_accuracy_Nt     
        balanced_accuracy_im[Nt]=balanced_accuracy_im_Nt
        balanced_accuracy_imtr[Nt]=balanced_accuracy_imtr_Nt
        

    balanced_accuracy['Domain']=source_domain
    balanced_accuracy_im['Domain']=target_domain
    balanced_accuracy_imtr['Domain']=target_domain+'_tr'
    results= pd.concat([balanced_accuracy,balanced_accuracy_im,balanced_accuracy_imtr])
    # results.to_csv(fname)
    print(f'Done {subject}: ',time()-t0)
else:
     print('Nothing done: The result file already exists.')

