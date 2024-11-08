# argument 1 is either NAIVE, RTLC
# argument 2 is either BASELINE, NAIVE
fsl_location=$FSLDIR/data/standard/MNI152_T1_2mm_brain_mask

python 1.NOcentered_maps_to_mni_DA.py $1 $fsl_location
python 1.NOcentered_maps_to_mni_DA.py $2 $fsl_location
ls -R ../../../results/searchlight/*/map_100iter_12mm_$1_nocentered_mni.nii.gz > list.txt
readarray -t files <list.txt
ls -R ../../../results/searchlight/*/map_100iter_12mm_$2_nocentered_mni.nii.gz > list2.txt
readarray -t filestosubstract < list2.txt
sed -e "s/${1}/${1}minus${2}/g" list.txt > list3.txt
readarray -t outfiles < list3.txt

counter=1

# get length of an array
length=${#files[@]}

# use for loop to read all values and indexes
for (( i=0; i<${length}; i++ ));
do
  fslmaths ${files[$i]} -sub ${filestosubstract[$i]} ${outfiles[$i]}
done
outfiles=$(cat list3.txt)
echo $outfiles
fslmerge -t ../../../results/searchlight/$1minus$2 $outfiles;

randomise -i ../../../results/searchlight/$1minus$2 -o ../../../results/searchlight/$1minus$2OneSampT -1 -v 6 -T -m $fsl_location -n 10000

