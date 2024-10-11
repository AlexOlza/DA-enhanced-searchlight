#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:40:07 2023

@author: alexolza

This script fits a DA method with data from a particular ROI and subject.
The DA method is then evaluated in the independent samples from target domain.
This process is repeated for NITER data partitions.
The underlying estimator is Logistic Regression.

Usage: python DA_comparison.py source_domain:str target_domain:str subject:int method:str region:int(-1 to use all regions) NITER:int
"""

""" THIRD PARTY IMPORTS """
import sys
import os
import re
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
from dataManipulation.loadDataDA import DomainAdaptationData, DomainAdaptationSplitter, DomainAdaptationGOD
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
def launch_priscilla():
    for m in method_names:
        for s in range(5):
            print(f'sbatch --job-name={s}{m} ../../main/raw.sh DA_comparison.py perception imagery {s} {m} VC 100 1')
#%%
def launch_priscilla_baseline():
        for s in range(5):
            print(f'sbatch --job-name={s}base ../../main/raw.sh baseline.py perception imagery {s} VC 100 0 1')
            print(f'sbatch --job-name={s}naive ../../main/raw.sh baseline.py perception imagery {s} VC 100 1 1')
#%%
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
    region = sys.argv[5]
    region_name = region
subject = subjects[ int(eval(sys.argv[3]))]

methods = [PRED, FA, SA,
           KMM, ULSIF, RULSIF, NearestNeighborsWeighting, IWN, BalancedWeighting, TrAdaBoost, RegularTransferLC]
method_names = [m.__name__ for m in methods]
methods = {n:m for n,m in zip(method_names,methods)}
parameters = {m : {} for m in method_names}


subject = subjects[ int(eval(sys.argv[3]))]
method = sys.argv[4]
NITER = int(eval(sys.argv[6]))


splitting='StratifiedGroupKFold'
n_folds = 5


fulldf=pd.DataFrame()

if dataset =='own':
    outdir = os.path.join('../results/DA_comparison', region_name, f'{source_domain}_{target_domain}', subject)
else:
    outdir = os.path.join(f'../results/DA_comparison/{dataset}', region_name, f'{source_domain}_{target_domain}', subject)
if not os.path.exists(outdir):
	os.makedirs(outdir)
""" MAIN PROGRAM """
#%%

t0=time()
params_base = {} if dataset=='own' else {'multi_class':'ovr', 'n_jobs':-1}
estimator = LogisticRegression(**params_base)

results={}
t0=time()

assert method in method_names, f'Unrecognized DA method {method}. Available methods: {method_names}. \n Usage: python DA_comparison.py source_domain:str target_domain:str subject:int method:str region:int(-1 to use all regions) NITER:int'

DA_method =  methods[method]
params = parameters[method]


print(f'Fitting {method} for subject {subject} in region {region_name} using dataset {dataset}...')
Nts=range(10,110, 10) if dataset=='own' else [200, 250, 300, 350, 400]

if not Path(os.path.join(outdir, f'DA_{method}.csv')).is_file():
    remove_noise=True
    Source_X, Source_y, Source_g = MyFullDataset(source_domain, subject, region, remove_noise=remove_noise,dataset=dataset)[:]
    Target_X, Target_y, Target_g = MyFullDataset(target_domain, subject, region, remove_noise=remove_noise,dataset=dataset)[:]
    
    # if dataset== 'ds001246': NITER = len(np.unique(Source_g))
    # 
    balanced_accuracy, balanced_accuracy_im, balanced_accuracy_imtr=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    for Nt in tqdm(Nts):
        balanced_accuracy_s,balanced_accuracy_im_s, balanced_accuracy_imtr_s=[],[],[]
        if dataset=='own':
            s = DomainAdaptationSplitter(StratifiedGroupKFold, NITER) 
            Source, Target=s.split(Source_X, Source_y, Source_g,Target_X, Target_y, Target_g,Nt, Nt)# Last argument is the random seed.
        else:
            Source, Target = DomainAdaptationGOD(Source_X, Target_X, Source_y, Target_y, Source_g, Target_g).split()
        d = DomainAdaptationData(Source, Target) #Just a wrapping class for convenience.
        prediction_fname =os.path.join(outdir, f'{method}_preds_{Nt}.csv')
        prediction_matrix =-1 *np.ones((len(Target_y),NITER))
        for i in range(NITER):
            
            train = d.Source_train_X[i]
            test = d.Source_test_X[i]

    
            train_label = np.ravel(d.Source_train_y[i])  
            test_label = np.ravel(d.Source_test_y[i])
            # We select a number "Nt" of instances from the target domain (usually Targetery)
            I_train, I_test, IL_train, IL_test = d.Target_train_X[i], d.Target_test_X[i], d.Target_train_y[i], d.Target_test_y[i]
            # I_train contains "Nt" instances. Those are passed to the ADAPT method
            I_test_idx =d.Target_test_i[i]
            
            if method=='RegularTransferLC':
            	# Parameter-based methods from the ADAPT library require an estimator that has been previously fit to the source domain
                estimator.fit(train,train_label)
    
                clf = DA_method( estimator,verbose=0,Xt=I_train,yt=IL_train, **params)   # clf recieves "Nt" instances of the target domain         
                clf.fit(I_train, IL_train)
            else:
                clf = DA_method(estimator = estimator,verbose=0,Xt=I_train,yt=IL_train, **params)   
         
                clf.fit(train,train_label,Xt=I_train,yt=IL_train)               # Feature and instance based methods perform the fitting and adaptation in one step
                								# so the estimator must not be previously fitted to the source domain
                                                     
            
            if method not in  ['TrAdaBoost', 'RegularTransferLC']:
                aux_ys = clf.predict(test, domain='src')                 # Predictions in source domain 
                aux_ys_Target = clf.predict(I_test, domain = 'tgt')        # Predictions in target domain
                aux_ys_Target_tr = clf.predict(I_train, domain = 'tgt')
            else:
                 aux_ys = clf.predict(test)                 # Predictions in source domain 
                 aux_ys_Target = clf.predict(I_test)          # Predictions in target domain
                 aux_ys_Target_tr = clf.predict(I_train)
                
            balanced_accuracy_s.append(balanced_accuracy_score( test_label, aux_ys))
            
            balanced_accuracy_im_s.append(balanced_accuracy_score( IL_test, aux_ys_Target))
            balanced_accuracy_imtr_s.append(balanced_accuracy_score( IL_train, aux_ys_Target_tr))
            prediction_matrix[I_test_idx,i] = aux_ys_Target
        np.save(prediction_fname,prediction_matrix)
        balanced_accuracy[Nt]=balanced_accuracy_s      
        balanced_accuracy_im[Nt]=balanced_accuracy_im_s
        balanced_accuracy_imtr[Nt]=balanced_accuracy_imtr_s
    
    balanced_accuracy['Domain']=source_domain
    balanced_accuracy_im['Domain']=target_domain
    balanced_accuracy_imtr['Domain']=target_domain+'_tr'
    results= pd.concat([balanced_accuracy,balanced_accuracy_im,balanced_accuracy_imtr])
    results.to_csv(os.path.join(outdir, f'DA_{method}.csv'))
    
    print(f'Done {subject}: ',time()-t0)

else:
    print('Nothing done: The result file already exists.')
