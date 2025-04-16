dataset=$1
N_classes=$2
NITER=100

if [ "$dataset" -eq "0" ]; then
	n_subj=18
	region=-1
else
	n_subj=5
	region=VC
fi

for (( subj=0; subj<$n_subj; subj++ )); # for each subject (only first n_subj done here)
do
	sbatch --job-name="n${N_classes}BASE" \
       --out="./slurm_logs/BASE_s${subj}data${dataset}n${N_classes}.out" \
       --error="./slurm_logs/BASE_s${subj}data${dataset}n${N_classes}.err" \
       --cpus-per-task=4 \
       ../send_sh_job.sh baseline.py imagery imagery $subj 0 $region $NITER $dataset $N_classes #with_tgt=0: baseline
       
       total_jobs=$(($total_jobs + 1))
done

