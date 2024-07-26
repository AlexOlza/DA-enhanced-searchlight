import sys
sys.path.append('../..')
from DA_comparison.evaluate.gather_results_DA_comparison import read_baseline, read_all
import pandas as pd
import os
import glob
import re
from pathlib import Path

if __name__=='__main__':
	OUTPATH = '../../results/RTLC'
	allregions=sorted([R.split('/')[-1].split('.')[0] for R in glob.glob(OUTPATH + '/*') if os.path.isdir(R)])
	subjects = sorted([S.split('/')[-1] for S in glob.glob(os.path.join(OUTPATH,f'{allregions[0]}/perception_imagery','*'))])
	fname = os.path.join(OUTPATH,'allresults_individualrois.csv')
	source_domain = 'perception'
	target_domain='imagery' 
	
	allresults= pd.DataFrame()
	base1= pd.DataFrame()
	for region_name in allregions:
	    baseline_region = read_baseline(region_name, OUTPATH='../../results/DA_comparison')
	    baseline_region = baseline_region.loc[baseline_region.Domain!=target_domain+'_tr']
	    baseline_region['Region']= region_name
	    base1= pd.concat([base1, baseline_region])
	base= pd.DataFrame()
	for region_name in allregions:
	    results = read_all( region_name, OUTPATH=OUTPATH)
	    results = results.loc[results.Method.isin(['RTLC','RTLC_LR'])]
	    results = results.loc[results.Domain!=target_domain+'_tr']
	    baseline_region = read_baseline(region_name, withtgt='', OUTPATH='../../results/DA_comparison')
	    baseline_region = baseline_region.loc[baseline_region.Domain!=target_domain+'_tr']
	    results['Region']= region_name
	    baseline_region['Region']= region_name
	    allresults = pd.concat([allresults,results])
	    base= pd.concat([base, baseline_region])

	base1['Method'] = 'NAIVE'
	base['Method'] = 'BASELINE'

	full_results = pd.concat([allresults,base,base1])
	full_results.to_csv(fname)
