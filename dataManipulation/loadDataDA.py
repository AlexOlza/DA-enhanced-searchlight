#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:37:53 2023

@author: alexolza
"""
from sklearn.model_selection import  train_test_split, LeavePGroupsOut, GroupShuffleSplit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DomainAdaptationData:
    def __init__(self, Source, Target):
        self.Source_train_X= Source['train_X']
        self.Source_test_X= Source['test_X'] 
        self.Source_train_y= Source['train_y']
        self.Source_test_y= Source['test_y']
        self.Source_train_g= Source['train_g']
        self.Source_test_g= Source['test_g']
        self.Source_train_i= Source['train_i']
        self.Source_test_i= Source['test_i']
        # self.Source_X= Source['X']
        # self.Source_y= Source['y']
        # self.Source_groups= Source['g']
        
        self.Target_train_X= Target['train_X']
        self.Target_test_X= Target['test_X'] 
        self.Target_train_y= Target['train_y']
        self.Target_test_y= Target['test_y']
        self.Target_train_g= Target['train_g']
        self.Target_test_g= Target['test_g']
        self.Target_train_i= Target['train_i']
        self.Target_test_i= Target['test_i']
        # self.Target_X= Target['X']
        # self.Target_y= Target['y']
        # self.Target_groups= Target['g']

class DomainAdaptationGOD:
    def __init__(self,Source_X, Target_X, source_events, target_events, source_groups, target_groups, n_iter=100, target_n=100, random_state = 0,
                          stratify_tgt=True):
        
        Source_train_is,Target_train_is,Source_test_is,Target_test_is = ds001246_cv_DA(source_events, target_events, source_groups, target_groups, n_iter, target_n, random_state,
                              stratify_tgt)
        self.Source_X=Source_X
        self.Target_X=Target_X
        self.Source_y=source_events
        self.Target_y=target_events
        self.Source_g=source_groups
        self.Target_g=target_groups
        self.Source_train_is=Source_train_is
        self.Target_train_is=Target_train_is
        self. Source_test_is=Source_test_is
        self.Target_test_is=Target_test_is
    def split(self):
        
        
        Source = {'train_X': self.Source_X[ self.Source_train_is],
                  'test_X': self.Source_X[ self.Source_test_is],
                  'train_y': self.Source_y[ self.Source_train_is],
                  'test_y': self.Source_y[ self.Source_test_is],
                  'train_g': self.Source_g[ self.Source_train_is],
                  'test_g': self.Source_g[ self.Source_test_is],
                  'train_i': self.Source_train_is,
                  'test_i': self.Source_test_is,
                  'X': self.Source_X,
                  'y': self.Source_y,
                  'g': self.Source_g}
        
        Target = {'train_X': self.Target_X[ self.Target_train_is],
                  'test_X': self.Target_X[ self.Target_test_is],
                  'train_y': self.Target_y[ self.Target_train_is],
                  'test_y': self.Target_y[ self.Target_test_is],
                  'train_g': self.Target_g[ self.Target_train_is],
                  'test_g': self.Target_g[ self.Target_test_is],
                  'train_i': self.Target_train_is,
                  'test_i': self.Target_test_is,
                  'X': self.Target_X,
                  'y': self.Target_y,
                  'g': self.Target_g}
        return(Source, Target)

class DomainAdaptationSplitter:
    def __init__(self, cv, n_iter):
        self.__name__='DomainAdaptationSplitter'
        self.n_iter = n_iter
        self.cv = cv #In our experimental design, this will be StratifiedKFold
        self.Source_X = None
        self.Source_y = None
        self.Source_groups = None
        self.Target_X = None
        self.Target_y = None
        self.Target_groups = None
        self.Target_n = None
        
    def split(self, Source_X, Source_y, Source_groups, Target_X, Target_y, Target_groups, Target_n, random_state=None):
        # This will return a dictionary with arrays of lenthg self.n_iter as values.
        # Each array contains the self.n_iter data partitions to be used.
        self.Source_X =Source_X
        self.Source_y =Source_y
        self.Source_groups =Source_groups
        self.Target_X =Target_X
        self.Target_y =Target_y 
        self.Target_groups =Target_groups
        self.Target_n =Target_n 
        
        Source_train_Xs, Source_test_Xs, Source_train_ys, Source_test_ys, Source_train_gs, Source_test_gs =[],[],[],[],[],[]
        Target_train_Xs, Target_test_Xs, Target_train_ys, Target_test_ys, Target_train_gs, Target_test_gs =  [],[],[],[],[],[]
        Source_test_is, Target_test_is, Source_train_is, Target_train_is =[],[],[],[]
        for i in range(self.n_iter):
            Source_train_index, Source_test_index = next(self.cv(n_splits=5, shuffle=True, random_state=i).split(Source_X, Source_y, groups=Source_groups))
            Source_train_X, Source_train_y = Source_X[Source_train_index], Source_y[Source_train_index]
            Source_test_X,Source_test_y = Source_X[Source_test_index], Source_y[Source_test_index]
                
            Target_train_index, Target_test_index =next( self.cv(n_splits=2, shuffle=True, random_state=i).split(Target_X, Target_y, groups=Target_groups))
            
            
            Final_Target_train_index, _ = train_test_split(Target_train_index, train_size=Target_n,stratify=Target_y[Target_train_index], random_state=random_state)
            
            Target_train_X,Target_train_y = Target_X[Final_Target_train_index], Target_y[Final_Target_train_index]
            
            groups_to_discard = Target_groups[Final_Target_train_index]
            indexes_to_append_to_test = [i for i in list(set(Final_Target_train_index)-set(Target_train_index)) if Target_groups[i] not in groups_to_discard]
            Final_Target_test_index = list(Target_test_index) + list(indexes_to_append_to_test)
            
            Target_test_X,Target_test_y = Target_X[Final_Target_test_index], Target_y[Final_Target_test_index]
            
            Target_train_g, Target_test_g = Target_groups[Final_Target_train_index], Target_groups[Final_Target_test_index]
            Source_train_g, Source_test_g = Source_groups[Source_train_index], Source_groups[Source_test_index]
            
            # These assertions guarantee no data leackage for our experimental design.
            assert len(set(Source_test_g).intersection(set(Source_train_g)))==0 
            assert len(set(Target_test_g).intersection(set(Target_train_g)))==0 
            
            Source_train_Xs.append(Source_train_X); Source_test_Xs.append(Source_test_X)
            Source_train_ys.append(Source_train_y); Source_test_ys.append(Source_test_y)
            
            Target_train_Xs.append(Target_train_X); Target_test_Xs.append(Target_test_X)
            Target_train_ys.append(Target_train_y); Target_test_ys.append(Target_test_y)
            
            Source_train_gs.append(Source_train_g); Source_test_gs.append(Source_test_g)
            Target_train_gs.append(Target_train_g); Target_test_gs.append(Target_test_g)
            
            Source_train_is.append(Source_train_index); Source_test_is.append(Source_test_index)
            Target_train_is.append(Final_Target_train_index); Target_test_is.append(Final_Target_test_index)
            
            

        Source = {'train_X': Source_train_Xs,
                  'test_X': Source_test_Xs,
                  'train_y': Source_train_ys,
                  'test_y': Source_test_ys,
                  'train_g': Source_train_gs,
                  'test_g': Source_test_gs,
                  'train_i': Source_train_is,
                  'test_i': Source_test_is,
                  'X': Source_X,
                  'y': Source_y,
                  'g':Source_groups}
        
        Target = {'train_X': Target_train_Xs,
                  'test_X': Target_test_Xs,
                  'train_y': Target_train_ys,
                  'test_y': Target_test_ys,
                  'train_g': Target_train_gs,
                  'test_g': Target_test_gs,
                  'train_i': Target_train_is,
                  'test_i': Target_test_is,
                  'X': Target_X,
                  'y': Target_y,
                  'g':Target_groups}
        self.Source= Source
        self.Target = Target
        
        
        
        return (Source,Target)            
    def get_n_splits(self, X, y, groups=None):
        return self.n_iter

def ds001246_cv_DA(source_events, target_events, source_groups, target_groups, n_iter=100, target_n=100, random_state = 0,
                      stratify_tgt=True):
    Source_test_is, Target_test_is, Source_train_is, Target_train_is =[],[],[],[]
    splitter = LeavePGroupsOut(5)
    # print((len(target_events)-target_n), target_n)
    tgt_splitter = LeavePGroupsOut(2) #GroupShuffleSplit(n_splits=n_iter, train_size=(len(target_events)-target_n))
    src_splits=list(splitter.split(source_events,groups=source_groups))
    tgt_splits = list(tgt_splitter.split(target_events,groups=target_groups))
    # n_iter =min( len(src_splits),len(tgt_splits))
    for i in range(n_iter):
        Source_train_index, Source_test_index = src_splits[i]
        Target_train_index, Target_test_index = tgt_splits[i]
        
        stratify = target_events[Target_train_index] if stratify_tgt else None
        
        Final_Target_train_index, _ = train_test_split(Target_train_index, train_size=target_n, random_state=i,stratify=stratify)

        groups_to_discard = target_groups[Final_Target_train_index]

        indexes_to_append_to_test = [i for i in list(set(Final_Target_train_index)-set(Target_train_index)) if target_groups[i] not in groups_to_discard]
        Final_Target_test_index = list(Target_test_index) + list(indexes_to_append_to_test)

        Target_train_g, Target_test_g = target_groups[Final_Target_train_index], target_groups[Final_Target_test_index]
        Source_train_g, Source_test_g = source_groups[Source_train_index], source_groups[Source_test_index]
        
        assert len(set(Source_test_g).intersection(set(Source_train_g)))==0 
        assert len(set(Target_test_g).intersection(set(Target_train_g)))==0 
        

        
        Source_train_is.append(Source_train_index); Source_test_is.append(Source_test_index)
        Target_train_is.append(Final_Target_train_index); Target_test_is.append(Final_Target_test_index)
        
    return(Source_train_is,Target_train_is,Source_test_is,Target_test_is)


#%%