# @author:alexolza

dataset=$1
N_classes=$2
REDUCE_DIM=$3
NITER=100

if [ "$dataset" -eq "0" ]; then
	n_subj=18
	region=-1
else
	n_subj=5
	region=VC
fi

total_jobs=0

declare -a methods=("DeepCORAL" "DANN" "MCD" "FineTuning")

## DL comparison
for method in "${methods[@]}"
do
   echo $method
   for (( subj=0; subj<$n_subj; subj++ )); # for each subject (only first n_subj done here)
	do
	sbatch --job-name="n${N_classes}${method}" \
       --out="./slurm_logs/${method}_s${subj}data${dataset}n${N_classes}dim${dim_reduction}.out" \
       --error="./slurm_logs/${method}_s${subj}data${dataset}n${N_classes}dim${dim_reduction}.err" \
       --cpus-per-task=16 \
       ../send_sh_job.sh DA_comparison.py perception imagery $subj $method $region $NITER $dataset $N_classes $REDUCE_DIM
       total_jobs=$(($total_jobs + 1))
	done	
done

echo 'SENT TOTAL NUMBER OF JOBS: ', $total_jobs


