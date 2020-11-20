#!/bin/bash

output_dir=$1
pref_name=$2
if [ -d $output_dir ]
then
  rm $output_dir
fi

ner_data_folder_pref="/home/aakdemir/biobert_data/datasets/subsetNER_for_QAS_1000_*"
root_folder="/home/aakdemir/"
dataset_folders=$(ls ${ner_data_folder_pref})
# Fix the dataset for now
for subset_folder in $dataset_folders
do
    echo $subset_folder
    subset_folder_path=$root_folder "/" $subset_folder
    datasets=$(ls ${ner_data_folder_pref})
    for target_fd in $datasets
    do
      folder_path=$subset_folder_path"/"$target_fd
      echo "Dataset path "$folder_path
    done
done

