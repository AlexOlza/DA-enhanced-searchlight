
from nibabel import load as load_fmri
import numpy as np
import pandas as pd
import os
import re
from nilearn.maskers import NiftiMasker
from nilearn.image import index_img

def load_data(domain, subject, rootpath='..' , drop_resting_state=True): 
    # Loads the complete whole-brain dataset for a domain and subject
    data_dir = f'{rootpath}/data/whole_brain/{domain}/{subject}/'
    event_dir = f'{rootpath}/data/{domain}/{subject}/events.csv'
    events = pd.read_csv(event_dir, usecols=['trial_idx','run','target_category'])
    bold_data = load_fmri(os.path.join(data_dir,f'{domain}_detrended_zscored_stacked_vols_of_interest_searchlight.nii.gz'))
    if drop_resting_state:
	    white_noise_events_idx = np.where(events.target_category==2)[0]
	    events = events.drop(white_noise_events_idx)
	    # I discard the scans corresponding to white noise
	    bold_data = index_img(bold_data,events.index)
    
    example_func = os.path.join(re.sub(domain,'perception',data_dir),'example_func_deoblique_brainmask.nii')

    ex_f = load_fmri(example_func)
    masker = NiftiMasker(ex_f).fit()
    
    return(bold_data, events, masker)
