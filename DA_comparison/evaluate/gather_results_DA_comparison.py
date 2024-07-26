import sys
sys.path.append('../..')
OUTPATH = '../../results/DA_comparison'
import pandas as pd
import os
import glob
import re
from pathlib import Path
def isupper(string): return(re.sub('[a-z]','',string))
def read_baseline(region_name='all_regions', source_domain='perception',target_domain='imagery', withtgt='_withtgt', OUTPATH=OUTPATH):
    baseline=pd.DataFrame()
    path = os.path.join(OUTPATH, region_name, f'{source_domain}_{target_domain}', '*')
    Subjects = [s.split('/')[-1] for s in glob.glob(path)] 
    
    for subject in Subjects: 
        result_path = os.path.join(OUTPATH, region_name, f'{source_domain}_{target_domain}', subject, f'baseline{withtgt}.csv')
        results = pd.read_csv(result_path, index_col=0)
        results['Subject']=subject
        baseline=pd.concat([baseline,results])   
    return baseline
def read_all(region_name='all_regions', source_domain='perception',target_domain='imagery', OUTPATH=OUTPATH):
    allresults=pd.DataFrame()
    
    path_ = os.path.join(OUTPATH, region_name, f'{source_domain}_{target_domain}', '*')
    methods = [ ] 
    for m in glob.glob(path_+'/*.csv'):
        try: methods.append(re.search(f'DA_(.*).csv',m.split('/')[-1]).group(1))
        except AttributeError: pass
    methods = list(set(methods))
    for m in methods:
        path = os.path.join(path_, f'DA_{m}.csv')
        Subjects = [s.split('/')[-2] for s in glob.glob(path)] 
        for subject in Subjects: 
            result_path = os.path.join(OUTPATH, region_name,  f'{source_domain}_{target_domain}', subject, f'DA_{m}.csv')
            results = pd.read_csv(result_path, index_col=0)
            results['Subject']=subject
            results['Method']=m
            allresults=pd.concat([allresults,results])
    allresults.Method = allresults.Method.apply(lambda m: isupper(m))  
    return allresults
if __name__=='__main__':
	fname = os.path.join(OUTPATH,'allresults_DA_comparison.csv')
	subjects = sorted([S.split('/')[-1] for S in glob.glob(os.path.join(OUTPATH,'all_regions/perception_imagery/*'))])
	source_domain = 'perception'
	target_domain='imagery' 
	naive = read_baseline()
	results = read_all()
	results = results.loc[results.Domain!=target_domain+'_tr']
	baseline = read_baseline( withtgt='')
	naive['Method']='NAIVE'
	baseline['Method']='BASELINE'
	full_results = pd.concat([results,baseline,naive])
	full_results.to_csv(fname)
