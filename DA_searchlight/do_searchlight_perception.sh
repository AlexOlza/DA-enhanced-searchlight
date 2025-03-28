#!/bin/bash
for (( i=0; i<18; i++ )); # for each subject (only first n_subj done here)
do
echo 'SUBJECT :' $i
for (( j=0; j<4; j++ )); # for each subject (only first n_subj done here)
do
echo 'radius index :' $j
sbatch ../send_sh_job.sh $1 $i 100 $2 $j #1 is the searchlight python script, 2 is 0 or 1 for the DA algorithm
done
done
