# @author:alexolza

dataset=$1
region=$2
dim_reduction=$3
NITER=100
n_subj=18
n_reg=14

declare -a methods=("DeepCORAL" "DANN" "MCD" "FineTuning")

## now loop through the above array
for method in "${methods[@]}"
do
   echo $method
   for (( subj=0; subj<$n_subj; subj++ )); # for each subject (only first n_subj done here)
	do
	sbatch --job-name="s${subj}dr${dim_reduction}${method}" \
       --out="./slurm_logs/${method}_s${subj}data${dataset}dim${dim_reduction}.out" \
       --error="./slurm_logs/${method}_s${subj}data${dataset}dim${dim_reduction}.err" \
       ../send_sh_job.sh DA_comparison_DL.py perception imagery $subj $region $method $NITER $dataset $dim_reduction
	done	
done



