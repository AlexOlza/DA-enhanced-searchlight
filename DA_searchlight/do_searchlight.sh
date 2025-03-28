for (( i=2; i<18; i++ )); # for each subject (only first n_subj done here)
do
echo 'SUBJECT :' $i
python searchlight.py $i 100 $1 $2
done
