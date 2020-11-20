#!/bin/bash

output_dir=$1
pref_name=$2
if [ -d $output_dir ]
then
  rm $output_dir
fi

ner_data_folder_pref="/home/aakdemir/biobert_data/datasets/subsetNER_for_QAS_1000_*"
dataset_folders=$(ls ${ner_data_folder_pref})
# Fix the dataset for now
for subset_folder in $dataset_folders
do
    echo $subset_folder
    subset_folder_path=$ner_data_folder_pref "/" $subset_folder
    datasets=$(ls ${ner_data_folder_pref})
    for target_fd $datasets
    do
      folder_path=$subset_folder_path"/"$target_fd
      echo "NER FOLDER "$folder_path
    done
done

