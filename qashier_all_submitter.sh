#!/bin/bash

output_dir=$1
ner_folder=$2
ner_data_folder=$3
pref_name=$4
if [ -d $output_dir ]
then
  rm $output_dir
fi

datasets=$(ls ${ner_data_folder})
# Fix the dataset for now
target=0
for target_fd in $datasets
do
    echo "Submitting "${target_fd}
    folder_path=$ner_data_folder"/"$target_fd
    qsub -N $pref_name"_nermodel_"$target_fd /home/aakdemir/bioMLT/qashier_submit.sh $output_dir $ner_folder $target_fd
done

