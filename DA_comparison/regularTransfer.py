#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:41:18 2023

@author: alexolza
Usage: python regularTransfer.py source_domain:str target_domain:str subject:int region:int(-1 to use all regions) NITER:int
"""

""" THIRD PARTY IMPORTS """
import sys
import os
sys.path.append("..")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from time import time
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from adapt.parameter_based import RegularTransferLR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
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

subjects = sorted(
    [S.split("/")[-1] for S in glob.glob(os.path.join("../data", "perception", "*"))]
)
allregions = sorted(
    [
        R.split("/")[-1].split(".")[0]
        for R in glob.glob(os.path.join("../data", "perception/1", "*.npy"))
    ]
)

source_domain = sys.argv[1]
target_domain = sys.argv[2]
subject = subjects[int(eval(sys.argv[3]))]
region = (
    allregions if int(eval(sys.argv[4])) == -1 else allregions[int(eval(sys.argv[4]))]
)
NITER = int(eval(sys.argv[5]))
splitting = "StratifiedGroupKFold"
n_folds = 5
region_name = "all_regions" if int(eval(sys.argv[5])) == -1 else region

fulldf = pd.DataFrame()

outdir = os.path.join(
    "../results/RTLC", region_name, f"{source_domain}_{target_domain}", subject
)
if not os.path.exists(outdir):
    os.makedirs(outdir)
# %%

""" MAIN PROGRAM """
# %%
DA_method = RegularTransferLC
method = "RTLC"
fname = os.path.join(outdir, f"DA_{method}.csv")
if not Path(fname).is_file():

    results = {}
    t0 = time()


    print(method)
    x = range(10, 110, 10)

    remove_noise = True

    perc_X, perc_y, perc_g = MyFullDataset(
        source_domain,
        subject,
        region,
        remove_noise=remove_noise,
    )[:]
    imag_X, imag_y, imag_g = MyFullDataset(
        target_domain, subject, region, remove_noise=remove_noise
    )[:]

    estimator = LogisticRegression()


    balanced_accuracy, balanced_accuracy_im, balanced_accuracy_imtr = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    for shots in tqdm(x):
        balanced_accuracy_s, balanced_accuracy_im_s, balanced_accuracy_imtr_s = [], [], []
        s = DomainAdaptationSplitter(StratifiedGroupKFold, NITER)
        Source, Target = s.split(
            perc_X, perc_y, perc_g, imag_X, imag_y, imag_g, shots, shots
        )  # last arg is random seed
        d = DomainAdaptationData(Source, Target)
        for i in range(NITER):

            train = d.Source_train_X[i]
            test = d.Source_test_X[i]

            train_label = np.ravel(d.Source_train_y[i])
            test_label = np.ravel(d.Source_test_y[i])
            # We select a number "shots" of instances from the target domain (usually imagery)
            I_train, I_test, IL_train, IL_test = (
                d.Target_train_X[i],
                d.Target_test_X[i],
                d.Target_train_y[i],
                d.Target_test_y[i],
            )

            estimator.fit(train, train_label)

            clf = DA_method(
                estimator, verbose=0, Xt=I_train, yt=IL_train
            )  # PRED recieves "shots" instances of the target domain
            clf.fit(I_train, IL_train, verbose=0)

            aux_ys = estimator.predict(test)  # Predictions in source domain
            aux_ys_imag = clf.predict(I_test)  # Predictions in target domain
            aux_ys_imag_tr = clf.predict(I_train)

            balanced_accuracy_s.append(balanced_accuracy_score(test_label, aux_ys))

            balanced_accuracy_im_s.append(balanced_accuracy_score(IL_test, aux_ys_imag))
            balanced_accuracy_imtr_s.append(
                balanced_accuracy_score(IL_train, aux_ys_imag_tr)
            )

        balanced_accuracy[shots] = balanced_accuracy_s
        balanced_accuracy_im[shots] = balanced_accuracy_im_s
        balanced_accuracy_imtr[shots] = balanced_accuracy_imtr_s

    balanced_accuracy["Domain"] = source_domain
    balanced_accuracy_im["Domain"] = target_domain
    balanced_accuracy_imtr["Domain"] = target_domain + "_tr"
    results = pd.concat([balanced_accuracy, balanced_accuracy_im, balanced_accuracy_imtr])
    results.to_csv(fname)

    print(f"Done {subject}: ", time() - t0)
else:
     print('Nothing done: The result file already exists.')
