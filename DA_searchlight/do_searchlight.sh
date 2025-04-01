min_suj=$(( $4 ))
max_suj=$(( $5 ))
echo ' running script: ' $1
echo 'algorithm: ' $2
for (( i=$min_suj; i<$max_suj; i++ )); # for each subject (only first n_subj done here)
do
echo 'SUBJECT AND RADIUS:' $i ' AND ' $3
#python searchlight.py $i 100 $1 $2
sbatch --output="slurm_logs/%x_alg${2}_subj${i}_rad${3}.out" --error="slurm_logs/%x_alg${2}_subj${i}_rad${3}.err" ../send_sh_job.sh $1 $i 100 $2 $3 #1 is the searchlight python script, 2 is 0 or 1 for the DA algorithm, 3 for radius
done
