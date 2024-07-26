#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alexolza

Contains functions that run a DA-enhanced searchlight in parallel.

"""


from nilearn import image
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
from joblib import Parallel, cpu_count, delayed
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity
from nilearn.image.resampling import coord_transform
from nilearn.decoding.searchlight import search_light, GroupIterator,_group_iter_search_light

def searchlight_cv_DA(source_events, target_events, source_groups, target_groups, n_iter=100, target_n=100, random_state = 0, splitter=StratifiedGroupKFold):
    # 
    Source_train_ys, Source_test_ys, Source_train_gs, Source_test_gs =[],[],[],[]
    Target_train_ys, Target_test_ys, Target_train_gs, Target_test_gs =  [],[],[],[]
    Source_test_is, Target_test_is, Source_train_is, Target_train_is =[],[],[],[]
    target_groups.index = range(len(target_groups))
    source_groups.index = range(len(source_groups))
    for i in range(n_iter):
        Source_train_index, Source_test_index = next(splitter(n_splits=5, shuffle=True, random_state=i).split(range(len(source_events)),source_events, groups=source_groups))
        Source_train_y = source_events[Source_train_index]
        Source_test_y = source_events[Source_test_index]
            
        Target_train_index, Target_test_index =next( splitter(n_splits=2, shuffle=True, random_state=i).split(range(len(target_events)),target_events, groups=target_groups))
        
        
        Final_Target_train_index, _ = train_test_split(Target_train_index, train_size=target_n,stratify=target_events[Target_train_index], random_state=i)
        
        Target_train_y =  target_events[Final_Target_train_index]
        
        groups_to_discard = target_groups[Final_Target_train_index]

        indexes_to_append_to_test = [i for i in list(set(Final_Target_train_index)-set(Target_train_index)) if target_groups[i] not in groups_to_discard]
        Final_Target_test_index = list(Target_test_index) + list(indexes_to_append_to_test)
        # print(Final_Target_test_index)
        Target_test_y = target_events[Final_Target_test_index]
        
        Target_train_g, Target_test_g = target_groups[Final_Target_train_index], target_groups[Final_Target_test_index]
        Source_train_g, Source_test_g = source_groups[Source_train_index], source_groups[Source_test_index]
        
        assert len(set(Source_test_g).intersection(set(Source_train_g)))==0 
        assert len(set(Target_test_g).intersection(set(Target_train_g)))==0 
        
        # Source_train_ys.append(Source_train_y); Source_test_ys.append(Source_test_y)
        
        # Target_train_ys.append(Target_train_y); Target_test_ys.append(Target_test_y)
        
        # Source_train_gs.append(Source_train_g); Source_test_gs.append(Source_test_g)
        # Target_train_gs.append(Target_train_g); Target_test_gs.append(Target_test_g)
        
        Source_train_is.append(Source_train_index); Source_test_is.append(Source_test_index)
        Target_train_is.append(Final_Target_train_index); Target_test_is.append(Final_Target_test_index)
        
    return(Source_train_is,Target_train_is)

def get_sphere_data(mask,BOLD_data,radius,allow_overlap=True):
    """
    inputs
    ---
    mask: the mask used for constraining the among of the brain we apply the searchlight algorithm
    BOLD_data: this doesn't even need to be the BOLD signals for everything, but just a few trials
            IMPORTANT: BOLD_data must be already loaded using nilearn.image.load_img
    radius: radius of the searchlight ball, in millimeters.

    return
    ---
    X: copy of BOLD_data
    A: **A.rows stores all the indicies for the moving sphere**
    process_mask: not important
    process_mask_affine: not important
    process_mask_coords: not important
    """
    if type(BOLD_data) == str:
        BOLD_data = image.load_img(BOLD_data)
    # Compute world coordinates of the seeds
    # process_mask_coords, process_mask_affine = masking.apply_mask(BOLD_data,mask) # masking has no attr apply mask
    process_mask_coords, process_mask_affine = mask.mask_img.get_fdata(), mask.affine_
    process_mask_coords = np.where(np.array(process_mask_coords)!=0)
    process_mask_coords = coord_transform(process_mask_coords[0],
                                          process_mask_coords[1],
                                          process_mask_coords[2],
                                          process_mask_affine,
                                          )
    process_mask_coords = np.asarray(process_mask_coords).T

    # seeds: list of triplets (x,y,z) containing the coordinates of each voxel (the centers of the spheres)
    X, A = _apply_mask_and_get_affinity(seeds = process_mask_coords, 
                                        niimg = BOLD_data,
                                        radius = radius,
                                        allow_overlap = allow_overlap,
                                        mask_img = mask.mask_img,
                                        )
    return X,A,process_mask_affine,process_mask_coords

from nilearn.image import new_img_like
from nilearn.maskers import NiftiLabelsMasker
def get_cluster_data(clusters_mask_img,BOLD_data):
    """
    inputs
    ---
    mask: the mask used for constraining the among of the brain we apply the searchlight algorithm
    BOLD_data: this doesn't even need to be the BOLD signals for everything, but just a few trials
            IMPORTANT: BOLD_data must be already loaded using nilearn.image.load_img
    radius: radius of the searchlight ball, in millimeters.

    return
    ---
    X: copy of BOLD_data
    A: **A.rows stores all the indicies for the moving sphere**
    process_mask: not important
    process_mask_affine: not important
    process_mask_coords: not important
    """
    if type(BOLD_data) == str:
       BOLD_data = image.load_img(BOLD_data)
    labels = clusters_mask_img.get_fdata()
    labels_img = new_img_like(BOLD_data, labels)
    # First, initialize masker with parameters suited for data extraction using
    # labels as input image, resampling_target is None as affine,
    # shape/size is same
    # for all the data used here, time series signal processing parameters
    # standardize and detrend are set to False
    masker = NiftiLabelsMasker(
        clusters_mask_img, resampling_target=None, standardize=False, detrend=False
    )
    # After initialization of masker object, we call fit() for preparing labels_img
    # data according to given parameters
    masker.fit()
    # Preparing for data extraction: setting number of conditions, size, etc from
    # haxby dataset
    condition_names = range(1,int(np.max(labels))+1)
    n_cond_img = BOLD_data.get_fdata()[ labels == 1,:].shape[-1]
    n_conds = len(condition_names)
    rows_c_min,rows_c_max = 0, 0
    Affinity=[]
    Data=[]
    for c in condition_names:
        n_cond_img = BOLD_data.get_fdata()[ labels == c,:].T
        rows_c_max += n_cond_img.shape[-1]
        Affinity.append([i for i in range(rows_c_min,rows_c_max)])
        rows_c_min = rows_c_max 
        Data.append(n_cond_img)
        
    return Data, Affinity


def search_light(
    X,
    y,
    estimator,
    A,
    X_test,
    y_test,
    groups=None,
    scoring=None,
    DA=None,
    X_tgt=None,
    y_tgt=None,
    n_jobs=-1,
    verbose=0,
):
    """Compute a search_light.

    Parameters
    ----------
    X : array-like of shape at least 2D
        data to fit.

    y : array-like
        target variable to predict.

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    A : scipy sparse matrix.
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.

    groups : array-like, optional, (default None)
        group label for each sample for cross validation.

        .. note::
            This will have no effect for scikit learn < 0.18

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        for possible values.
        If callable, it takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.
    %(n_jobs_all)s
    %(verbose0)s
    
    DA : ADAPT method, optional
        An ADAPT method. If None, no domain adaptation is performed
        and X_tgt, y_tgt are ignored.
        
    X_tgt : array-like of shape at least 2D, optional
        Instances from the target domain to perform domain adaptation on.
        Must be supplied if a DA method is supplied.
    
    y_tgt : array-like, optional
        Labels from the target domain to perform domain adaptation on.
        Must be supplied if a SUPERVISED DA method is supplied.

    Returns
    -------
    scores : array-like of shape (number of rows in A)
        search_light scores
    """
    n_jobs = cpu_count()-2 if n_jobs==-1 else n_jobs
    group_iter = GroupIterator(A.shape[0], n_jobs)
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_group_iter_search_light)(
                A.rows[list_i],
                estimator,
                X,
                y,
                groups,
                scoring,
                X_test,
                y_test,
                DA,
                X_tgt,
                y_tgt,
                thread_id + 1,
                A.shape[0],
                verbose,
            )
            for thread_id, list_i in enumerate(group_iter)
        )
    return np.concatenate(scores)



def _group_iter_search_light(
    list_rows,
    estimator,
    X,
    y,
    groups,
    scoring,
    X_test,
    y_test,
    DA,
    X_tgt,
    y_tgt,
    thread_id,
    total,
    verbose=0,
):
    """Perform grouped iterations of search_light.

    Parameters
    ----------
    list_rows : array of arrays of int
        adjacency rows. For a voxel with index i in X, list_rows[i] is the list
        of neighboring voxels indices (in X).

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    X : array-like of shape at least 2D
        data to fit.

    y : array-like
        target variable to predict.

    groups : array-like, optional
        group label for each sample for cross validation.

    scoring : string or callable, optional
        Scoring strategy to use. See the scikit-learn documentation.
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross validation is
        used or 3-fold stratified cross-validation when y is supplied.

    thread_id : int
        process id, used for display.

    total : int
        Total number of voxels, used for display

    verbose : int, optional
        The verbosity level. Default is 0

    Returns
    -------
    par_scores : numpy.ndarray
        score for each voxel. dtype: float64.
    """
    par_scores = np.zeros(len(list_rows))
    t0 = time.time()
    for i, row in enumerate(list_rows):
        kwargs = {"scoring": scoring, "groups": groups}
        fitted_estimator = estimator().fit(X[:, row], y)
        assert hasattr(fitted_estimator, 'coef_')
        if DA is not None:
            # print('Performing domain adaptation')
            clf = DA(fitted_estimator,Xt=X_tgt[:, row],yt=y_tgt).fit(X_tgt[:, row], y_tgt)
            y_pred = clf.predict(X_test[:, row])
            # return(y_pred)
        else:
            y_pred = fitted_estimator.predict(X_test[:, row])
        
        par_scores[i] = scoring(y_test, y_pred)

        if verbose > 0:
            # One can't print less than each 10 iterations
            step = 11 - min(verbose, 10)
            if i % step == 0:
                # If there is only one job, progress information is fixed
                crlf = "\r" if total == len(list_rows) else "\n"
                percent = float(i) / len(list_rows)
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                remaining = (100.0 - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    f"Job #{thread_id}, processed {i}/{len(list_rows)} voxels "
                    f"({percent:0.2f}%, {remaining} seconds remaining){crlf}"
                )
    return par_scores

