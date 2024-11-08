# argument 1 is either BASELINE, NAIVE, RTLC, PERCEPTION
# argument 2 is NITER
fsl_location=$FSLDIR/data/standard/MNI152_T1_2mm_brain_mask
ls -R ../../../results/searchlight/*/map_$2iter_12mm_$1_centered_mni.nii.gz > list.txt

files=$(cat list.txt)
echo "$files"
fslmerge -t ../../../results/searchlight/$1 $files;

randomise -i ../../../results/searchlight/$1 -o $1OneSampT -1 -v 6 -T -m $fsl_location.nii.gz -n 10000
mv $1OneSampT_tfce_corrp_tstat1.nii.gz ../../../results/searchlight/$1OneSampT_tfce_corrp_tstat1.nii.gz
mv $1OneSampT_tstat1.nii.gz ../../../results/searchlight/$1OneSampT_tstat1.nii.gz

