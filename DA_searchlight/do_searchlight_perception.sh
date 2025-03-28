for (( i=2; i<18; i++ )); # for each subject (only first n_subj done here)
do
echo 'SUBJECT :' $i
python perception_searchlight.py $i 100 $1
done
