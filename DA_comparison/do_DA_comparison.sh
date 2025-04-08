# @author:alexolza

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

declare -a methods=("PRED" "FA" "SA" "KMM" "ULSIF" "RULSIF" "NearestNeighborsWeighting" "IWN" "BalancedWeighting" "TrAdaBoost" "RegularTransferLC")

total_jobs=0

## BASELINE and NAIVE
for (( subj=0; subj<$n_subj; subj++ )); # for each subject (only first n_subj done here)
do
	sbatch --job-name="n${N_classes}BASE" \
       --out="./slurm_logs/BASE_s${subj}data${dataset}n${N_classes}.out" \
       --error="./slurm_logs/BASE_s${subj}data${dataset}n${N_classes}.err" \
       --cpus-per-task=4 \
       ../send_sh_job.sh baseline.py perception imagery $subj 0 $region $NITER $dataset $N_classes #with_tgt=0: baseline
       
       sbatch --job-name="n${N_classes}NAIVE" \
       --out="./slurm_logs/NAIVE_s${subj}data${dataset}n${N_classes}.out" \
       --error="./slurm_logs/NAIVE_s${subj}data${dataset}n${N_classes}.err" \
       --cpus-per-task=4 \
       ../send_sh_job.sh baseline.py perception imagery $subj 1 $region $NITER $dataset $N_classes #with_tgt=1: naive 
       
       total_jobs=$(($total_jobs + 1))
done

## DA comparison
for method in "${methods[@]}"
do
   echo $method
   for (( subj=0; subj<$n_subj; subj++ )); # for each subject (only first n_subj done here)
	do
	sbatch --job-name="n${N_classes}${method}" \
       --out="./slurm_logs/${method}_s${subj}data${dataset}n${N_classes}.out" \
       --error="./slurm_logs/${method}_s${subj}data${dataset}n${N_classes}.err" \
       --cpus-per-task=4 \
       ../send_sh_job.sh DA_comparison.py perception imagery $subj $method $region $NITER $dataset $N_classes 
       total_jobs=$(($total_jobs + 1))
	done	
done

echo 'SENT TOTAL NUMBER OF JOBS: ', $total_jobs


