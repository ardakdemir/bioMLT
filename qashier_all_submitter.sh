#!/bin/bash

output_dir=$1
ner_folder=$2
pref_name=$3
if [ -d $output_dir ]
then
  rm $output_dir
fi

ner_data_folder="/home/aakdemir/biobert_data/datasets/NER"
datasets=$(ls ${ner_data_folder})
# Fix the dataset for now
target=0
for target_fd in $datasets
do
    echo "Submitting "${target_fd}
    folder_path=$ner_data_folder"/"$target_fd
    qsub -N $pref_name"_nermodel_"$target_fd /home/aakdemir/bioMLT/qashier_submit.sh $output_dir"_"${target_fd} $ner_folder $target_fd
done

