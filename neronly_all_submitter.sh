#!/bin/bash

output_dir=$1
pref_name=$2
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
    folder_path=$ner_data_folder"/"$target_fd
    qsub -N $pref_name"_target_"$target_fd /home/aakdemir/bioMLT/neronly_mtl_singledataset_submit.sh $output_dir $target_fd
done

