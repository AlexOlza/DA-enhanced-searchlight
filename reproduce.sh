# @author:alexolza

# This script reproduces all the figures from the paper 
# "Domain Adaptation-Enhanced Searchlight: Enabling brain decoding from visual perception to mental imagery" (Olza A., Soto D., Santana R., 2024).
# The real experimentation was heavily parallelized.

echo "Reproducing Domain Adaptation-Enhanced Searchlight: Enabling brain decoding from visual perception to mental imagery"
echo "(Olza A., Soto D., Santana R., 2024)"
echo "Visit https://github.com/AlexOlza/DA-enhanced-searchlight for more information"
echo "Expected CPU time is around 33 h with NITER=1"
# Paper parameters: NITER=100, n_subj=18, n_reg=14
NITER=1
n_subj=18
n_reg=14
fsl_location=$FSLDIR/data/standard/MNI152_T1_2mm_brain_mask
cd DA_comparison
for (( i=0; i<$n_subj; i++ )); # for each subject (only first n_subj done here)
do
echo 'SUBJECT :' $i
# Fit each DA method in the union of all ROIs (region=-1)
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
for (( j=0; j<$n_reg; j++ )); # for each individual region, fit RTLC and the baseline
do
python regularTransfer.py perception imagery $i $j $NITER
python baseline.py perception imagery $i $j $NITER 0
python baseline.py perception imagery $i $j $NITER 1
done
done
cd evaluate
# Make two .csv files with the results
python gather_results_DA_comparison.py
python gather_results_individual_ROIs.py
echo 'CRITICAL DIFFERENCE DIAGRAMS...'
# Make Figures 1, 2, 3, 4, 5, and A.10
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
python searchlight_frequency_analysis.py $NITER # Make Figure 8
echo 'SEARCHLIGHT NORMALISATION...'
cd searchlight_normalisation/scripts
# Substract the chance level accuracy
python 0.mean_centering_searchlight_maps_DA.py BASELINE $NITER
python 0.mean_centering_searchlight_maps_DA.py NAIVE $NITER
python 0.mean_centering_searchlight_maps_DA.py RTLC $NITER
python 0.mean_centering_searchlight_maps_DA.py PERCEPTION $NITER

# Convert to MNI space
python 1.centered_maps_to_mni_DA.py BASELINE $fsl_location
python 1.NOcentered_maps_to_mni_DA.py BASELINE $fsl_location
python 1.centered_maps_to_mni_DA.py NAIVE $fsl_location
python 1.NOcentered_maps_to_mni_DA.py NAIVE $fsl_location
python 1.centered_maps_to_mni_DA.py RTLC $fsl_location
python 1.NOcentered_maps_to_mni_DA.py RTLC $fsl_location
python 1.centered_maps_to_mni_DA.py PERCEPTION $fsl_location
python 1.NOcentered_maps_to_mni_DA.py PERCEPTION $fsl_location

# Perform intra-subject statistical analysis
python 2.subjectwise_variability_analysis.py BASELINE $NITER $fsl_location
python 2.subjectwise_variability_analysis.py NAIVE $NITER $fsl_location
python 2.subjectwise_variability_analysis.py RTLC $NITER $fsl_location
python 2.subjectwise_variability_analysis.py PERCEPTION $NITER $fsl_location

# Merge the MNI centered maps from different subjects and perform cross-subject statistical analysis
source 3.merge_centered_randomise.sh BASELINE $NITER $fsl_location
source 3.merge_centered_randomise.sh NAIVE $NITER $fsl_location
source 3.merge_centered_randomise.sh RTLC $NITER $fsl_location
source 3.merge_centered_randomise.sh PERCEPTION $NITER $fsl_location

python 4.global_average_to_significant_clusters.py # Project the average accuracy of each voxel onto the significant clusters
python 5.nilearn_plots.py # Make Figure 6
# Make figures 7, A.11, A.12 and A.13
python 6.subject_variability.py BASELINE 
python 6.subject_variability.py NAIVE
python 6.subject_variability.py RTLC
python 6.subject_variability.py PERCEPTION
