#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:40:07 2023

@author: alexolza

This script fits a DA method with data from a particular ROI and subject.
The DA method is then evaluated in the independent samples from target domain.
This process is repeated for NITER data partitions.
The underlying estimator is Logistic Regression.

Usage: python DA_comparison_DL.py source_domain:str target_domain:str subject:int method:str region:int(-1 to use all regions) NITER:int
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
import tensorflow as tf
from adapt.feature_based import DeepCORAL, DANN, MCD
from adapt.parameter_based import FineTuning
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import warnings
from sklearn.decomposition import FastICA
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

""" OWN IMPORTS """

from dataManipulation.loadDataDA import DomainAdaptationData, DomainAdaptationSplitter
from dataManipulation.loadData import MyFullDataset

#%%
""" VARIABLE DEFINITION """

subjects = sorted([S.split('/')[-1] for S in glob.glob(os.path.join('../data','perception','*'))])
allregions=sorted([R.split('/')[-1].split('.')[0] for R in glob.glob(os.path.join(f'../data/perception/{subjects[0]}','*.npy'))])
methods = [DeepCORAL, DANN, MCD, FineTuning ]
method_names = [m.__name__ for m in methods]
methods = {n:m for n,m in zip(method_names,methods)}
parameters = {m : {} for m in method_names}

source_domain =  sys.argv[1]
target_domain =  sys.argv[2]
subject = subjects[ int(eval(sys.argv[3]))]
method = sys.argv[4]
region = allregions if int(eval(sys.argv[5]))==-1 else allregions[int(eval(sys.argv[5]))]
NITER = int(eval(sys.argv[6]))
splitting='StratifiedGroupKFold'
n_folds = 5
region_name='all_regions' if int(eval(sys.argv[5]))==-1 else region

def launch_priscilla():
    i=0
    for s in range(1,19):
            for m, method in enumerate(method_names):
                print(f'sbatch --job-name={s}{method[:3]} ../../main/raw.sh DA_comparison_DL.py perception imagery {s} {method} -1 100')
                i+=1
    print('Total number of jobs: ', i)


fulldf=pd.DataFrame()

outdir = os.path.join('../results/DA_comparison', region_name, f'{source_domain}_{target_domain}', subject)
if not os.path.exists(outdir):
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass
""" MAIN PROGRAM """
#%%

results={}
t0=time()

assert method in method_names, f'Unrecognized DA method {method}. Available methods: {method_names}. \n Usage: python DA_comparison.py source_domain:str target_domain:str subject:int method:str region:int(-1 to use all regions) NITER:int'

DA_method =  methods[method]
params = parameters[method]


print(f'Fitting {method} for subject {subject} in region {region_name}...')
Nts=range(10,110, 10)

if not Path(os.path.join(outdir, f'DA_{method}.csv')).is_file():
    remove_noise=True
    perc_X, perc_y, perc_g = MyFullDataset(source_domain, subject, region, remove_noise=remove_noise)[:]
    imag_X, imag_y, imag_g = MyFullDataset(target_domain, subject, region, remove_noise=remove_noise)[:]
    
    
    
    balanced_accuracy, balanced_accuracy_im, balanced_accuracy_imtr=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    for Nt in tqdm(Nts):
        random_state=Nt
        balanced_accuracy_s,balanced_accuracy_im_s, balanced_accuracy_imtr_s=[],[],[]
        s = DomainAdaptationSplitter(StratifiedGroupKFold, NITER)
        Source, Target=s.split(perc_X, perc_y, perc_g,imag_X, imag_y, imag_g,Nt,random_state)# last arg is random seed
        d = DomainAdaptationData(Source, Target)
        for i in range(NITER):
            
            train = d.Source_train_X[i]
            test = d.Source_test_X[i]
            ica = FastICA(100).fit(train)
            train_ICA = ica.transform(train)
            test_ICA = ica.transform(test)
    
            train_label = np.ravel(d.Source_train_y[i])  
            test_label = np.ravel(d.Source_test_y[i])
            # We select a number "Nt" of instances from the target domain (usually imagery)
            I_train, I_test, IL_train, IL_test = d.Target_train_X[i], d.Target_test_X[i], d.Target_train_y[i], d.Target_test_y[i]
            # I_train contains "Nt" instances. Those are passed to the ADAPT method
            
            ica_tgt = FastICA(100).fit(I_train)
            I_train_ICA = ica.transform(I_train)
            I_test_ICA = ica.transform(I_test)
            
            n_features = 100
            if method=='FineTuning':
                FineTuning_model = FineTuning(None, None,train_ICA, train_label)
                FineTuning_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                   optimizer=tf.keras.optimizers.legacy.Adam(0.001), 
                                   metrics=["accuracy"]
                                   )
                                   
                # Train FineTuning
                FineTuning_model.fit(train_ICA, train_label, batch_size=128, epochs=100,
                                     validation_data=(I_train_ICA,IL_train),verbose=0
                              )

                #%%

                clf = FineTuning(encoder=FineTuning_model.encoder_, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                       pretrain=True,  pretrain__epochs=30, random_state=random_state,
                                       optimizer=tf.keras.optimizers.legacy.Adam(0.001),
                                       optimizer_enc=tf.keras.optimizers.legacy.Adam(0.0001),
                                       metrics = ['accuracy'])

                clf.fit(I_train_ICA,IL_train, epochs=100, verbose=0)


                # print("Evaluate")
                # result = clf.predict(I_train_ICA)
                # pred_classes = np.where(result.ravel()>=0.5,1,0)
                # print(balanced_accuracy_score(pred_classes, IL_train))
                # result = clf.predict(I_test_ICA)
                # pred_classes = np.where(result.ravel()>=0.5,1,0)
                # print(balanced_accuracy_score(pred_classes, IL_test))
                # import pandas as pd
                # from matplotlib import pyplot as plt
                # pd.DataFrame(clf.history_).plot(figsize=(8, 5))
                # plt.title("Training history", fontsize=14); plt.xlabel("Epochs"); plt.ylabel("Scores")
                # plt.legend(ncol=2)
                # plt.show()
            elif method=='DANN':
                clf = DANN()
                clf.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                    optimizer=tf.keras.optimizers.legacy.Adam(0.001),
                                    optimizer_enc=tf.keras.optimizers.legacy.Adam(0.0001),
                                    metrics=["accuracy"])
                                   
                # Train DANN
                clf.fit(X=train_ICA,y=train_label, Xt=I_train_ICA ,batch_size=64, epochs=500,
                             )

            elif method=='DeepCORAL':
                clf = DeepCORAL(Xt=I_train_ICA, metrics=["accuracy"],
                                  optimizer=tf.keras.optimizers.legacy.Adam(0.001),
                                  optimizer_enc=tf.keras.optimizers.legacy.Adam(0.0001),
                                  random_state=random_state)

                clf.fit(X=train_ICA,y=train_label, epochs=200, verbose=0)
            elif method=='MCD':
                clf = MCD(Xt=I_train_ICA, metrics=["accuracy"],
                                  optimizer=tf.keras.optimizers.legacy.Adam(0.001),
                                  optimizer_enc=tf.keras.optimizers.legacy.Adam(0.0001),
                                  random_state=random_state)

                clf.fit(X=train_ICA,y=train_label, epochs=200, verbose=1)  
            else:
                assert False, 'Unrecognised DA method'     
            
           
            aux_ys = np.where(clf.predict(test_ICA).ravel()>=0.5 ,1,0)               # Predictions in source domain 
            aux_ys_imag = np.where(clf.predict(I_test_ICA).ravel()>=0.5 ,1,0)         # Predictions in target domain
            aux_ys_imag_tr =np.where( clf.predict(I_train_ICA).ravel()>=0.5 ,1,0)
                
            balanced_accuracy_s.append(balanced_accuracy_score( test_label, aux_ys))
            
            balanced_accuracy_im_s.append(balanced_accuracy_score( IL_test, aux_ys_imag))
            balanced_accuracy_imtr_s.append(balanced_accuracy_score( IL_train, aux_ys_imag_tr))
            
        balanced_accuracy[Nt]=balanced_accuracy_s      
        balanced_accuracy_im[Nt]=balanced_accuracy_im_s
        balanced_accuracy_imtr[Nt]=balanced_accuracy_imtr_s
    
    balanced_accuracy['Domain']=source_domain
    balanced_accuracy_im['Domain']=target_domain
    balanced_accuracy_imtr['Domain']=target_domain+'_tr'
    results= pd.concat([balanced_accuracy,balanced_accuracy_im,balanced_accuracy_imtr])
    # results.to_csv(os.path.join(outdir, f'DA_{method}.csv'))
    
    print(f'Done {subject}: ',time()-t0)

else:
    print('Nothing done: The result file already exists.')

#%%
