NITER=100
n_subj=18
n_reg=14
fsl_location=$FSLDIR/data/standard/MNI152_T1_2mm_brain_mask
cd DA_comparison
for (( i=0; i<$n_subj; i++ )); # for each subject (only first n_subj done here)
do
echo 'SUBJECT :' $i
python DA_comparison.py perception imagery $i PRED -1 $NITER
python DA_comparison.py perception imagery $i RegularTransferLC -1 $NITER
python DA_comparison.py perception imagery $i FA -1 $NITER
python DA_comparison.py perception imagery $i TrAdaBoost -1 $NITER
python DA_comparison.py perception imagery $i BalancedWeighting -1 $NITER
python DA_comparison.py perception imagery $i RULSIF -1 $NITER
python DA_comparison.py perception imagery $i ULSIF -1 $NITER
python DA_comparison.py perception imagery $i KMM -1 $NITER
python DA_comparison.py perception imagery $i IWN -1 $NITER
python DA_comparison.py perception imagery $i NearestNeighborsWeighting -1 $NITER
python DA_comparison.py perception imagery $i SA -1 $NITER
python baseline.py perception imagery $i -1 $NITER 0
python baseline.py perception imagery $i -1 $NITER 1
for (( j=0; j<$n_reg; j++ )); # for each region
do
python regularTransfer.py perception imagery $i $j $NITER
python baseline.py perception imagery $i $j $NITER 0
python baseline.py perception imagery $i $j $NITER 1
done
done
cd evaluate
python gather_results_DA_comparison.py
python gather_results_individual_ROIs.py
echo 'CRITICAL DIFFERENCE DIAGRAMS...'

Rscript critical_difference_diagrams_indivROIs.R
Rscript critical_difference_diagrams.R
python heatmaps_CD_diagrams.py
echo 'SEARCHLIGHT...'
cd ../../DA_searchlight
for (( i=0; i<$n_subj; i++ )); # for each subject (only first n_subj done here)
do
echo 'SUBJECT :' $i
python perception_searchlight.py $i $NITER
python searchlight.py $i $NITER
done
python searchlight_frequency_analysis.py $NITER
echo 'SEARCHLIGHT NORMALISATION...'
cd searchlight_normalisation/scripts
python 0.mean_centering_searchlight_maps_DA.py BASELINE $NITER
python 0.mean_centering_searchlight_maps_DA.py NAIVE $NITER
python 0.mean_centering_searchlight_maps_DA.py RTLC $NITER
python 0.mean_centering_searchlight_maps_DA.py PERCEPTION $NITER

python 1.centered_maps_to_mni_DA.py BASELINE $fsl_location
python 1.NOcentered_maps_to_mni_DA.py BASELINE $fsl_location
python 1.centered_maps_to_mni_DA.py NAIVE $fsl_location
python 1.NOcentered_maps_to_mni_DA.py NAIVE $fsl_location
python 1.centered_maps_to_mni_DA.py RTLC $fsl_location
python 1.NOcentered_maps_to_mni_DA.py RTLC $fsl_location
python 1.centered_maps_to_mni_DA.py PERCEPTION $fsl_location
python 1.NOcentered_maps_to_mni_DA.py PERCEPTION $fsl_location

python 2.subjectwise_variability_analysis.py BASELINE $NITER $fsl_location
python 2.subjectwise_variability_analysis.py NAIVE $NITER $fsl_location
python 2.subjectwise_variability_analysis.py RTLC $NITER $fsl_location
python 2.subjectwise_variability_analysis.py PERCEPTION $NITER $fsl_location

source 3.merge_centered_randomise.sh BASELINE $NITER $fsl_location
source 3.merge_centered_randomise.sh NAIVE $NITER $fsl_location
source 3.merge_centered_randomise.sh RTLC $NITER $fsl_location
source 3.merge_centered_randomise.sh PERCEPTION $NITER $fsl_location

python 4.global_average_to_significant_clusters.py
python 5.nilearn_plots.py
python 6.subject_variability.py BASELINE
python 6.subject_variability.py NAIVE
python 6.subject_variability.py RTLC
python 6.subject_variability.py PERCEPTION
