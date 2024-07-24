#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:40:07 2023

@author: alexolza

Usage: python DA_comparison.py source_domain:str target_domain:str subject:int method:str region:int(-1 to use all regions) NITER:int
"""

""" THIRD PARTY IMPORTS """
import sys
import os
sys.path.append('..')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from time import time
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from adapt.feature_based import PRED, FA, CORAL, SA, fMMD
from adapt.instance_based import LDM, KLIEP, KMM, ULSIF, RULSIF, NearestNeighborsWeighting, IWC, IWN, BalancedWeighting, TrAdaBoost
from adapt.parameter_based import RegularTransferLR
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

""" OWN IMPORTS """

from dataManipulation.loadDataDA import DomainAdaptationData, DomainAdaptationSplitter
from dataManipulation.loadData import MyFullDataset
from adapt.utils import check_arrays
                         
from sklearn.preprocessing import LabelBinarizer
class RegularTransferLC(RegularTransferLR):
	def fit(self, Xt=None, yt=None, **fit_params):       
		Xt, yt = self._get_target_data(Xt, yt)
		Xt, yt = check_arrays(Xt, yt)
		
		_label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
		_label_binarizer.fit(self.estimator.classes_)
		yt = _label_binarizer.transform(yt)
		
		# print(yt.shape) -> this print is present in the current release of ADAPT, and it is very annoying
		
		return super().fit(Xt, yt, **fit_params)

#%%
""" VARIABLE DEFINITION """

subjects = sorted([S.split('/')[-1] for S in glob.glob(os.path.join('../data','perception','*'))])
allregions=sorted([R.split('/')[-1].split('.')[0] for R in glob.glob(os.path.join('../data','perception/1','*.npy'))])
methods = [PRED, FA, CORAL, SA, fMMD,
           LDM, KLIEP, KMM, ULSIF, RULSIF, NearestNeighborsWeighting, IWC, IWN, BalancedWeighting, TrAdaBoost, RegularTransferLC]
method_names = [m.__name__ for m in methods]
methods = {n:m for n,m in zip(method_names,methods)}
parameters = {m : {} for m in method_names}
parameters['IWC'] = {'classifier':LogisticRegression(), 'kernel':'polynomial'}   
parameters['KLIEP'] = {'kernel':'polynomial'}

source_domain =  sys.argv[1]
target_domain =  sys.argv[2]
subject = subjects[ int(eval(sys.argv[3]))]
method = sys.argv[4]
region = allregions if int(eval(sys.argv[5]))==-1 else allregions[int(eval(sys.argv[5]))]
NITER = int(eval(sys.argv[6]))
splitting='StratifiedGroupKFold'
n_folds = 5
region_name='all_regions' if int(eval(sys.argv[5]))==-1 else region


fulldf=pd.DataFrame()

outdir = os.path.join('../results/DA_comparison', region_name, f'{source_domain}_{target_domain}', subject)
if not os.path.exists(outdir):
	os.makedirs(outdir)
""" MAIN PROGRAM """
#%%

t0=time()

estimator = LogisticRegression()

results={}
t0=time()

assert method in method_names, f'Unrecognized DA method {method}. Available methods: {method_names}. \n Usage: python DA_comparison.py source_domain:str target_domain:str subject:int method:str region:int(-1 to use all regions) NITER:int'

DA_method =  methods[method]
params = parameters[method]


print(f'Fitting {method} for subject {subject} in region {region_name}...')
x=range(10,110, 10)

if not Path(os.path.join(outdir, f'DA_{method}.csv')).is_file():
    remove_noise=True
    perc_X, perc_y, perc_g = MyFullDataset(source_domain, subject, region, remove_noise=remove_noise)[:]
    imag_X, imag_y, imag_g = MyFullDataset(target_domain, subject, region, remove_noise=remove_noise)[:]
    
    
    
    balanced_accuracy, balanced_accuracy_im, balanced_accuracy_imtr=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    for shots in tqdm(x):
        balanced_accuracy_s,balanced_accuracy_im_s, balanced_accuracy_imtr_s=[],[],[]
        s = DomainAdaptationSplitter(StratifiedGroupKFold, NITER)
        Source, Target=s.split(perc_X, perc_y, perc_g,imag_X, imag_y, imag_g,shots,shots)# last arg is random seed
        d = DomainAdaptationData(Source, Target)
        for i in range(NITER):
            
            train = d.Source_train_X[i]
            test = d.Source_test_X[i]

    
            train_label = np.ravel(d.Source_train_y[i])  
            test_label = np.ravel(d.Source_test_y[i])
            # We select a number "shots" of instances from the target domain (usually imagery)
            I_train, I_test, IL_train, IL_test = d.Target_train_X[i], d.Target_test_X[i], d.Target_train_y[i], d.Target_test_y[i]
            # I_train contains "shots" instances. Those are passed to the ADAPT method
           
            
            if method=='RegularTransferLC':
            	# Parameter-based methods from the ADAPT library require an estimator that has been previously fit to the source domain
                estimator.fit(train,train_label)
    
                clf = DA_method( estimator,verbose=0,Xt=I_train,yt=IL_train, **params)   # clf recieves "shots" instances of the target domain         
                clf.fit(I_train, IL_train)
            else:
                clf = DA_method(estimator = estimator,verbose=0,Xt=I_train,yt=IL_train, **params)   
         
                clf.fit(train,train_label,Xt=I_train,yt=IL_train)               # Feature and instance based methods perform the fitting and adaptation in one step
                								# so the estimator must not be previously fitted to the source domain
                                                     
            
            if method not in  ['TrAdaBoost', 'RegularTransferLC']:
                aux_ys = clf.predict(test, domain='src')                 # Predictions in source domain 
                aux_ys_imag = clf.predict(I_test, domain = 'tgt')        # Predictions in target domain
                aux_ys_imag_tr = clf.predict(I_train, domain = 'tgt')
            else:
                 aux_ys = clf.predict(test)                 # Predictions in source domain 
                 aux_ys_imag = clf.predict(I_test)          # Predictions in target domain
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
    results.to_csv(os.path.join(outdir, f'DA_{method}.csv'))
    
    print(f'Done {subject}: ',time()-t0)

else:
    print('Nothing done: The result file already exists.')
