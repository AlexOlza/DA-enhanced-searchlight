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
os.environ['TF_USE_LEGACY_KERAS']='True'
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import warnings
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

""" OWN IMPORTS """
from dataManipulation.loadDataDA import DomainAdaptationData, DomainAdaptationSplitter, DomainAdaptationGOD, reduce_dim
from dataManipulation.loadData import MyFullDataset

#%%
# Disable prints, which are inside ADAPT fit methods and I can't disable them:
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
#%%

# Define networks
# from tensorflow.keras.layers import Dense, ReLU, Softmax, Activation, Flatten
# class SequentialModule(tf.keras.Model):
#   def __init__(self, input_dim, output_dim, output_activation=tf.keras.activations.sigmoid, name=None):
#     super().__init__(name=name)
#     self.input_dim = input_dim
#     self.output_dim=output_dim
#     self.dense_1 = Dense(100,input_dim=input_dim, activation='relu')
#     self.dense_2 = Dense(10, activation='relu')
#     self.output_layer = Dense(output_dim, activation=output_activation)
#   def __call__(self, x, training=False):
#     x = self.dense_1(x)
#     x = self.dense_2(x)
#     return self.output_layer(x)
  # def summary(self):
  #       super().summary(self)
# class Encoder_model(SequentialModule):
#     def __init__(self,input_dim, output_dim=10, output_activation=tf.keras.activations.relu, name='enc'):
#       super().__init__(input_dim, output_dim,output_activation, name)
# # DANN needs a discriminator. The documentation states that 
# # "the output shape of the discriminator should be (None, 1) and a sigmoid activation should be used."
# # (here: https://adapt-python.github.io/adapt/generated/adapt.feature_based.DANN.html)
# class Discriminator_model(SequentialModule):
#     def __init__(self,input_dim, output_dim=1, output_activation=tf.keras.activations.sigmoid, name='disc'):
#       super().__init__(input_dim, output_dim,output_activation, name)

# class Task_model(SequentialModule):
#     def __init__(self,input_dim=10, output_dim=1, output_activation=tf.keras.activations.relu, name='task'):
#       super().__init__(input_dim, output_dim,output_activation, name)

# def make_encoder_model(n_voxels):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.Input(shape=(n_voxels,)))  # more idiomatic than input_shape in first Dense
#     model.add(Dense(100))
#     model.add(ReLU())
#     model.add(Dense(10))
#     model.add(ReLU())
#     model.add(Flatten())
#     return model

# def make_discriminator_model(n_voxels):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.Input(shape=(n_voxels,)))  # more idiomatic than input_shape in first Dense
#     model.add(Dense(100))
#     model.add(ReLU())
#     model.add(Dense(10))
#     model.add(Dense(1))
#     model.add(Activation("sigmoid"))
#     return model

# def make_task_model(n_voxels):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.Input(shape=(n_voxels,)))  # more idiomatic than input_shape in first Dense
#     model.add(Dense(100))
#     model.add(ReLU())
#     model.add(Dense(10))
#     model.add(Softmax())
#     return model
#%%
""" VARIABLE DEFINITION """
source_domain =  sys.argv[1]
target_domain =  sys.argv[2]
dataset = int(eval(sys.argv[7]))
N_classes = int(eval(sys.argv[8]))
REDUCE_DIM = int(eval(sys.argv[9]))
shuffle = False
average=False
binary=False
oversample=False

dataset = ['own', 'ds0012146', 'ds001246_semantic'][dataset]
if dataset=='own':
    subjects = sorted([S.split('/')[-1] for S in glob.glob(os.path.join('../data','perception','*'))])
    allregions=sorted([R.split('/')[-1].split('.')[0] for R in glob.glob(os.path.join(f'../data/perception/{subjects[0]}','*.npy'))])
    idx = [int(r) for r in sys.argv[5].split('+')]
    region = allregions if int(eval(sys.argv[5]))==-1 else [allregions[i] for i in idx]  
    region_name='all_regions' if int(eval(sys.argv[5]))==-1 else '-'.join(region)
elif dataset =='ds001246':  
    subjects = sorted([S.split('/')[-1].split('.')[0] for S in glob.glob(os.path.join(f'../{dataset}','Subject[0-9].h5'))])
    region = sys.argv[5]
    region_name = region
elif dataset =='ds001246_semantic':
    subjects = sorted([S.split('/')[-1].split('.')[0] for S in glob.glob(os.path.join(f'../ds001246','Subject[0-9].h5'))])
    region = sys.argv[5]
    region_name = region

methods = [DeepCORAL, DANN, MCD, FineTuning ]
method_names = [m.__name__ for m in methods]
methods = {n:m for n,m in zip(method_names,methods)}
parameters = {m : {} for m in method_names}
living=False

subject = subjects[ int(eval(sys.argv[3]))]
method = sys.argv[4]
NITER_ = int(eval(sys.argv[6]))

splitting='StratifiedGroupKFold'
n_folds = 5
ICA=True

#%%
fulldf=pd.DataFrame()

if dataset =='own':
    outdir = os.path.join('../results/DA_comparison', region_name, f'{source_domain}_{target_domain}', subject)
elif dataset=='ds001246':
    outdir = os.path.join(f'../results/DA_comparison/{dataset}_resnet{N_classes}', region_name, f'{source_domain}_{target_domain}', subject)
else:
    outdir = os.path.join(f'../results/DA_comparison/{dataset}', region_name, f'{source_domain}_{target_domain}', subject)
if not os.path.exists(outdir):
	os.makedirs(outdir)
""" MAIN PROGRAM """
#%%

results={}
t0=time()

assert method in method_names, f'Unrecognized DA method {method}. Available methods: {method_names}. \n Usage: python DA_comparison.py source_domain:str target_domain:str subject:int method:str region:int(-1 to use all regions) NITER:int'

DA_method =  methods[method]
params = parameters[method]

REDUCE_DIM_NAME = ['NONE', 'TSVD', 'SRP', 'ICA'][REDUCE_DIM]
result_filename = os.path.join(outdir, f'DA_DL_{method}_reduce{REDUCE_DIM_NAME}.csv')
print('WILL PRODUCE FILE: ', result_filename)
print(f'Fitting {method} for subject {subject} in region {region_name}...')

Nts=range(10,110, 10) if dataset=='own' else [200, 250, 300]
if True:#not Path(result_filename).is_file():
    remove_noise=True
    Source_X, Source_y, Source_g = MyFullDataset(source_domain, subject, region, 
                                                 remove_noise=remove_noise,
                                                 dataset=dataset,
                                                 N_classes=N_classes
                                                 )[:]
    Target_X, Target_y, Target_g = MyFullDataset(target_domain, subject, region, 
                                                 remove_noise=remove_noise,
                                                 dataset=dataset,
                                                 N_classes=N_classes
                                                 )[:]
    
    balanced_accuracy, balanced_accuracy_im, balanced_accuracy_imtr=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
    for Nt in tqdm(Nts):
        random_state=Nt
        balanced_accuracy_s,balanced_accuracy_im_s, balanced_accuracy_imtr_s=-np.ones(NITER_),-np.ones(NITER_),-np.ones(NITER_)
        if dataset=='own':
            s = DomainAdaptationSplitter(StratifiedGroupKFold, NITER_) 
            Source, Target=s.split(Source_X, Source_y, Source_g,Target_X, Target_y, Target_g,Nt, Nt)# Last argument is the random seed.
        else:
            Source, Target = DomainAdaptationGOD(Source_X, Target_X, Source_y, Target_y, Source_g, Target_g).split()
        d = DomainAdaptationData(Source, Target) #Just a wrapping class for convenience.
        prediction_fname =os.path.join(outdir, f'DL_{method}_preds_{Nt}_reduce{REDUCE_DIM_NAME}')
        NITER =min(NITER_, len(d.Target_train_y))
        prediction_matrix =-1 *np.ones((len(Target_y),NITER))
        with HiddenPrints():
            for i in tqdm(range(NITER)):
                
                train = d.Source_train_X[i]
                test = d.Source_test_X[i]
                train_label = np.ravel(d.Source_train_y[i])  
                test_label = np.ravel(d.Source_test_y[i])
                
                # print('Original dataset shape %s' % Counter(train_label))
                # ros = RandomOverSampler(random_state=i)
                # if living:
                #     train_label[train_label!=1]=0
                #     test_label[test_label!=1]=0
                # # print('Original dataset shape %s' % Counter(train_label))
                # ros = RandomOverSampler(random_state=i)
               
                # if oversample: train, train_label = ros.fit_resample(train, train_label)
                # print('Resampled dataset shape %s' % Counter(train_label))
                # We select a number "Nt" of instances from the target domain (usually Targetery)
                tr_idx = d.Target_train_i[i]
                te_idx =  d.Target_test_i[i]
                I_train, I_test, IL_train, IL_test = Target_X[tr_idx], Target_X[te_idx], Target_y[tr_idx], Target_y[te_idx]
                # I_train contains "Nt" instances. Those are passed to the ADAPT method
                I_test_idx =d.Target_test_i[i]
                
                # if oversample: I_train, IL_train = ros.fit_resample(I_train, IL_train)
                # print('Resampled target dataset shape %s' % Counter(IL_train))
                
                if REDUCE_DIM!=0:
                    train, I_train, test, I_test = reduce_dim(train, I_train, test, I_test, n_components = 100, method=REDUCE_DIM)    
    
                n_voxels = train.shape[-1]
                
                if method=='FineTuning':
                    # encoder = Encoder_model(n_voxels,10)
                    # task = Task_model(n_voxels)
                    # task = get_task(units=N_classes)
                    # discriminator = Discriminator_model(n_voxels)
                    FineTuning_model = FineTuning(#encoder, task,
                                                  # task=task
                                                  # X=train, y=train_label
                                                  )
                    FineTuning_model.compile(
                        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                                       optimizer=tf.keras.optimizers.legacy.Adam(0.001), 
                                       metrics=["accuracy"]
                                       )
                                       
                    # Train FineTuning
                    FineTuning_model.fit(train, train_label, 
                                         # batch_size=64,
                                         epochs=100,
                                         # validation_data=(I_train,IL_train),
                                         verbose=0
                                  )
    
                    #%%
    
                    clf = FineTuning(encoder=FineTuning_model.encoder_,
                                     # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                           pretrain=False,#True,  
                                           pretrain__epochs=30, random_state=random_state,
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
                    # encoder = Encoder_model(n_voxels,10)
                    # task = Task_model(n_voxels)
                    # discriminator = Discriminator_model(n_voxels)
                    # encoder = make_encoder_model(n_voxels)
                    # discriminator = make_discriminator_model(n_voxels)
                    # task = make_task_model(n_voxels)
                    clf = DANN()#encoder=encoder, discriminator=discriminator)
                    clf.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                        optimizer=tf.keras.optimizers.legacy.Adam(0.001),
                                        optimizer_enc=tf.keras.optimizers.legacy.Adam(0.0001),
                                        metrics=["accuracy"])
                                       
                    # Train DANN
                    clf =clf.fit(X=train,y=train_label, Xt=I_train,batch_size=64, epochs=500,
                                verbose=0 )
    
    
                elif method=='DeepCORAL':
                    # encoder = Encoder_model(n_voxels,10)
                    # task = Task_model(n_voxels)
                    clf = DeepCORAL(#encoder, task,
                                    Xt=I_train, metrics=["accuracy"],
                                      optimizer=tf.keras.optimizers.legacy.Adam(0.001),
                                      optimizer_enc=tf.keras.optimizers.legacy.Adam(0.0001),
                                      random_state=random_state)
    
                    clf.fit(X=train,y=train_label, epochs=200, verbose=0)
    
                elif method=='MCD':
                    # def get_task(activation="sigmoid", units=N_classes):
                    #     model = tf.keras.Sequential()
                    #     model.add(Flatten())
                    #     model.add(Dense(10, activation="relu"))
                    #     model.add(Dense(10, activation="relu"))
                    #     model.add(Dense(units, activation=activation))
                    #     return model
                    # encoder = Encoder_model(n_voxels,10)
                    # task = Task_model(n_voxels, output_dim=N_classes)
                    # task = get_task(units=N_classes)#make_task_model(n_voxels)
                    # discriminator = Discriminator_model(n_voxels)#.build(1,n_voxels)
                    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                    clf = MCD(Xt=I_train, metrics=["accuracy"], 
                              # task=task,
                              # encoder = encoder,
                                  # loss='categorical_crossentropy',#loss,
                                      optimizer=tf.keras.optimizers.legacy.Adam(0.001),
                                      optimizer_enc=tf.keras.optimizers.legacy.Adam(0.0001),
                                      random_state=random_state)
    
                    clf.fit(X=train,y=train_label,
                            # Xt=I_train, yt=IL_train, 
                            epochs=100, verbose=0)  
                    
    
                else:
                    assert False, 'Unrecognised DA method'     
                
                train_enc = clf.encoder_.predict(train)
                # train_enc__ = clf.transform(train)
                I_train_enc = clf.encoder_.predict(I_train)
                test_enc = clf.encoder_.predict(test)
                I_test_enc = clf.encoder_.predict(I_test)
                
                lr = LogisticRegression().fit(train_enc, train_label)
                
                aux_ys = lr.predict(test_enc)             # Predictions in source domain 
                aux_ys_imag = lr.predict(I_test_enc)       # Predictions in target domain
                aux_ys_imag_tr = lr.predict(I_train_enc)
                # print(np.unique(aux_ys))
                balanced_accuracy_s[i]=balanced_accuracy_score( test_label, aux_ys)
                
                balanced_accuracy_im_s[i]=balanced_accuracy_score( IL_test, aux_ys_imag)
                balanced_accuracy_imtr_s[i]=balanced_accuracy_score( IL_train, aux_ys_imag_tr)
                
                prediction_matrix[I_test_idx,i] = aux_ys_imag
            np.save(prediction_fname,prediction_matrix)
            balanced_accuracy[Nt]=balanced_accuracy_s      
            balanced_accuracy_im[Nt]=balanced_accuracy_im_s
            balanced_accuracy_imtr[Nt]=balanced_accuracy_imtr_s
        
    balanced_accuracy['Domain']=source_domain
    balanced_accuracy_im['Domain']=target_domain
    balanced_accuracy_imtr['Domain']=target_domain+'_tr'
    results= pd.concat([balanced_accuracy,balanced_accuracy_im,balanced_accuracy_imtr])
    results.to_csv(result_filename)
    
    print(f'Done {subject}: ',time()-t0)

else:
    print('Nothing done: The result file already exists.')

