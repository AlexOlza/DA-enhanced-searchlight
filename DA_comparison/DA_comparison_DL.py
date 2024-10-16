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
from dataManipulation.loadDataDA import DomainAdaptationData, DomainAdaptationSplitter, DomainAdaptationGOD
from dataManipulation.loadData import MyFullDataset

#%%

# Define networks
from tensorflow.keras.layers import Dense
class SequentialModule(tf.keras.Model):
  def __init__(self, input_dim, output_dim, output_activation=tf.keras.activations.sigmoid, name=None):
    super().__init__(name=name)
    self.input_dim = input_dim
    self.output_dim=output_dim
    self.dense_1 = Dense(100,input_dim=input_dim, activation='relu')
    self.dense_2 = Dense(10, activation='relu')
    self.output_layer = Dense(output_dim, activation=output_activation)
  def __call__(self, x, training=False):
    x = self.dense_1(x)
    x = self.dense_2(x)
    return self.output_layer(x)
  # def summary(self):
  #       super().summary(self)
class Encoder_model(SequentialModule):
    def __init__(self,input_dim, output_dim=10, output_activation=tf.keras.activations.relu, name='enc'):
      super().__init__(input_dim, output_dim,output_activation, name)
# DANN needs a discriminator. The documentation states that 
# "the output shape of the discriminator should be (None, 1) and a sigmoid activation should be used."
# (here: https://adapt-python.github.io/adapt/generated/adapt.feature_based.DANN.html)
class Discriminator_model(SequentialModule):
    def __init__(self,input_dim, output_dim=1, output_activation=tf.keras.activations.sigmoid, name='disc'):
      super().__init__(input_dim, output_dim,output_activation, name)

class Task_model(SequentialModule):
    def __init__(self,input_dim=10, output_dim=1, output_activation=tf.keras.activations.relu, name='task'):
      super().__init__(input_dim, output_dim,output_activation, name)


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
    subjects = sorted([S.split('/')[-1] for S in glob.glob(os.path.join('../data','Sourceeption','*'))])
    allregions=sorted([R.split('/')[-1].split('.')[0] for R in glob.glob(os.path.join(f'../data/Sourceeption/{subjects[0]}','*.npy'))])
    idx = [int(r) for r in sys.argv[4].split('+')]
    region = allregions if int(eval(sys.argv[4]))==-1 else [allregions[i] for i in idx]
    
    region_name='all_regions' if int(eval(sys.argv[4]))==-1 else '-'.join(region)
else:
    dataset ='ds001246'
    subjects = sorted([S.split('/')[-1].split('.')[0] for S in glob.glob(os.path.join(f'../{dataset}','*'))])
    region = sys.argv[5]
    region_name = region
subject = subjects[ int(eval(sys.argv[3]))]

methods = [DeepCORAL, DANN, MCD, FineTuning ]
method_names = [m.__name__ for m in methods]
methods = {n:m for n,m in zip(method_names,methods)}
parameters = {m : {} for m in method_names}


method = sys.argv[4]
NITER = int(eval(sys.argv[6]))
splitting='StratifiedGroupKFold'
n_folds = 5
ICA=True
#%%
def launch_priscilla():
    i=0
    for s in range(5):
            for m, method in enumerate(method_names):
                print(f'sbatch --job-name={s}{method[:3]} ../../main/raw.sh DA_comparison_DL.py perception imagery {s} {method} VC 100 1' )
                i+=1
    print('Total number of jobs: ', i)
#%%
fulldf=pd.DataFrame()
if dataset =='own':
    outdir = os.path.join('../results/DA_comparison', region_name, f'{source_domain}_{target_domain}', subject)
else:
    outdir = os.path.join(f'../results/DA_comparison/{dataset}_grouped', region_name, f'{source_domain}_{target_domain}', subject)
if not os.path.exists(outdir):
    os.makedirs(outdir)
""" MAIN PROGRAM """
#%%

results={}
t0=time()

assert method in method_names, f'Unrecognized DA method {method}. Available methods: {method_names}. \n Usage: python DA_comparison.py source_domain:str target_domain:str subject:int method:str region:int(-1 to use all regions) NITER:int'

DA_method =  methods[method]
params = parameters[method]


print(f'Fitting {method} for subject {subject} in region {region_name}...')
Nts=range(10,110, 10) if dataset=='own' else [200, 250, 300, 350, 400]
if not Path(os.path.join(outdir, f'DA_{method}.csv')).is_file():
    remove_noise=True
    Source_X, Source_y, Source_g = MyFullDataset(source_domain, subject, region, remove_noise=remove_noise,dataset=dataset)[:]
    Target_X, Target_y, Target_g = MyFullDataset(target_domain, subject, region, remove_noise=remove_noise,dataset=dataset)[:]
    
    # if dataset== 'ds001246': NITER = len(np.unique(Source_g))
    
    balanced_accuracy, balanced_accuracy_im, balanced_accuracy_imtr=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
    for Nt in tqdm(Nts):
        random_state=Nt
        balanced_accuracy_s,balanced_accuracy_im_s, balanced_accuracy_imtr_s=[],[],[]
        if dataset=='own':
            s = DomainAdaptationSplitter(StratifiedGroupKFold, NITER) 
            Source, Target=s.split(Source_X, Source_y, Source_g,Target_X, Target_y, Target_g,Nt, Nt)# Last argument is the random seed.
        else:
            Source, Target = DomainAdaptationGOD(Source_X, Target_X, Source_y, Target_y, Source_g, Target_g).split()
        d = DomainAdaptationData(Source, Target) #Just a wrapping class for convenience.
        prediction_fname =os.path.join(outdir, f'{method}_preds_{Nt}.csv')
        prediction_matrix =-1 *np.ones((len(Target_y),NITER))
        for i in tqdm(range(NITER)):
            
            train = d.Source_train_X[i]
            test = d.Source_test_X[i]
            train_label = np.ravel(d.Source_train_y[i])  
            test_label = np.ravel(d.Source_test_y[i])
            # We select a number "Nt" of instances from the target domain (usually imagery)
            I_train, I_test, IL_train, IL_test = d.Target_train_X[i], d.Target_test_X[i], d.Target_train_y[i], d.Target_test_y[i]
            # I_train contains "Nt" instances. Those are passed to the ADAPT method
            I_test_idx =d.Target_test_i[i]
            if ICA:
                ica = FastICA(100).fit(train)
                train = ica.transform(train)
                test = ica.transform(test)
                ica_tgt = FastICA(100).fit(I_train)
                I_train = ica_tgt.transform(I_train)
                I_test = ica_tgt.transform(I_test)

            n_voxels = train.shape[-1]
           
            if method=='FineTuning':
                encoder = Encoder_model(n_voxels,10)
                task = Task_model(n_voxels)
                discriminator = Discriminator_model(n_voxels)
                FineTuning_model = FineTuning(encoder, task,train, train_label)
                FineTuning_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                   optimizer=tf.keras.optimizers.legacy.Adam(0.001), 
                                   metrics=["accuracy"]
                                   )
                                   
                # Train FineTuning
                FineTuning_model.fit(train, train_label, batch_size=64, epochs=100,
                                     validation_data=(I_train,IL_train),verbose=0
                              )
                # ft.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                #                    optimizer=tf.keras.optimizers.legacy.Adam(0.001), 
                #                    metrics=["accuracy"]
                #                    )
                                   
                # # Train FineTuning
                # ft.fit(train, train_label, batch_size=64, epochs=100,
                #                      validation_data=(I_train,IL_train),verbose=0
                #               )
                #%%

                clf = FineTuning(encoder=FineTuning_model.encoder_, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                       pretrain=True,  pretrain__epochs=30, random_state=random_state,
                                       optimizer=tf.keras.optimizers.legacy.Adam(0.001),
                                       optimizer_enc=tf.keras.optimizers.legacy.Adam(0.0001),
                                       metrics = ['accuracy'])

                clf.fit(I_train,IL_train, epochs=10, verbose=0)


                # print("Evaluate")
                # result = clf.predict(I_train)
                # pred_classes = np.where(result.ravel()>=0.5,1,0)
                # print(balanced_accuracy_score(pred_classes, IL_train))
                # result = clf.predict(I_test)
                # pred_classes = np.where(result.ravel()>=0.5,1,0)
                # print(balanced_accuracy_score(pred_classes, IL_test))
                # import pandas as pd
                # from matplotlib import pyplot as plt
                # pd.DataFrame(clf.history_).plot(figsize=(8, 5))
                # plt.title(f"Training history ({clf.count_params()} parameters)", fontsize=14); plt.xlabel("Epochs"); plt.ylabel("Scores")
                # plt.legend(ncol=2)
                # plt.show()
            elif method=='DANN':
                encoder = Encoder_model(n_voxels,10)
                task = Task_model(n_voxels)
                discriminator = Discriminator_model(n_voxels)
                clf = DANN(encoder, task,discriminator)
                clf.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                    optimizer=tf.keras.optimizers.legacy.Adam(0.001),
                                    optimizer_enc=tf.keras.optimizers.legacy.Adam(0.0001),
                                    metrics=["accuracy"])
                                   
                # Train DANN
                clf.fit(X=train,y=train_label, Xt=I_train,batch_size=64, epochs=500,
                            verbose=0 )
                # print("Evaluate")
                # result = clf.predict(I_train)
                # pred_classes = np.where(result.ravel()>=0.5,1,0)
                # print(balanced_accuracy_score(pred_classes, IL_train))
                # result = clf.predict(I_test)
                # pred_classes = np.where(result.ravel()>=0.5,1,0)
                # print(balanced_accuracy_score(pred_classes, IL_test))
                # import pandas as pd
                # from matplotlib import pyplot as plt
                # pd.DataFrame(clf.history_).plot(figsize=(8, 5))
                # plt.title("Training history", fontsize=14); plt.xlabel("Epochs"); plt.ylabel("Scores")
                # plt.legend(ncol=2)
                # plt.show()

            elif method=='DeepCORAL':
                encoder = Encoder_model(n_voxels,10)
                task = Task_model(n_voxels)
                clf = DeepCORAL(encoder, task,Xt=I_train, metrics=["accuracy"],
                                  optimizer=tf.keras.optimizers.legacy.Adam(0.001),
                                  optimizer_enc=tf.keras.optimizers.legacy.Adam(0.0001),
                                  random_state=random_state)

                clf.fit(X=train,y=train_label, epochs=200, verbose=0)
                # print("Evaluate")
                # result = clf.predict(I_train)
                # pred_classes = np.where(result.ravel()>=0.5,1,0)
                # print(balanced_accuracy_score(pred_classes, IL_train))
                # result = clf.predict(I_test)
                # pred_classes = np.where(result.ravel()>=0.5,1,0)
                # print(balanced_accuracy_score(pred_classes, IL_test))
                # import pandas as pd
                # from matplotlib import pyplot as plt
                # pd.DataFrame(clf.history_).plot(figsize=(8, 5))
                # plt.title("Training history", fontsize=14); plt.xlabel("Epochs"); plt.ylabel("Scores")
                # plt.legend(ncol=2)
                # plt.show()

            elif method=='MCD':
                encoder = Encoder_model(n_voxels,10)
                task = Task_model(n_voxels)
                discriminator = Discriminator_model(n_voxels)
                clf = MCD(encoder, task,Xt=I_train, metrics=["accuracy"],
                                  optimizer=tf.keras.optimizers.legacy.Adam(0.001),
                                  optimizer_enc=tf.keras.optimizers.legacy.Adam(0.0001),
                                  random_state=random_state)

                clf.fit(X=train,y=train_label, epochs=200, verbose=0)  
                # print("Evaluate")
                # result = clf.predict(I_train)
                # pred_classes = np.where(result.ravel()>=0.5,1,0)
                # print(balanced_accuracy_score(pred_classes, IL_train))
                # result = clf.predict(I_test)
                # pred_classes = np.where(result.ravel()>=0.5,1,0)
                # print(balanced_accuracy_score(pred_classes, IL_test))
                # import pandas as pd
                # from matplotlib import pyplot as plt
                # pd.DataFrame(clf.history_).plot(figsize=(8, 5))
                # plt.title(f"Training history ({clf.count_params()} parameters)", fontsize=14); plt.xlabel("Epochs"); plt.ylabel("Scores")
                # plt.legend(ncol=2)
                # plt.show()

            else:
                assert False, 'Unrecognised DA method'     
            
           
            aux_ys = np.where(clf.predict(test).ravel()>=0.5 ,1,0)               # Predictions in source domain 
            aux_ys_imag = np.where(clf.predict(I_test).ravel()>=0.5 ,1,0)         # Predictions in target domain
            aux_ys_imag_tr =np.where( clf.predict(I_train).ravel()>=0.5 ,1,0)
                
            balanced_accuracy_s.append(balanced_accuracy_score( test_label, aux_ys))
            
            balanced_accuracy_im_s.append(balanced_accuracy_score( IL_test, aux_ys_imag))
            balanced_accuracy_imtr_s.append(balanced_accuracy_score( IL_train, aux_ys_imag_tr))
            
            prediction_matrix[I_test_idx,i] = aux_ys_imag
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

#%%
