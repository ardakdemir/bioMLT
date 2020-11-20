#!/bin/bash

output_pref=$1
pref_name=$2
if [ -d $output_dir ]
then
  rm $output_dir
fi

ner_data_folder_pref="/home/aakdemir/biobert_data/datasets/subsetNER_for_QAS_*"
root_folder="/home/aakdemir/"
dataset_folders=$(ls -d ${ner_data_folder_pref})
# Fix the dataset for now
x=0
for subset_folder in $dataset_folders
do
    echo "Subset folder: "$subset_folder
    subset_folder_path=${subset_folder}
    datasets=$(ls ${subset_folder_path})
    exp_code=${$subset_folder:55}
    output_dir=$output_pref"_"$exp_code
    y=0
    for target_fd in $datasets
    do
      folder_path=$subset_folder_path"/"$target_fd
      echo "Dataset path "$folder_path
      echo "Output dir: "$output_dir
      y=$((y + 1))
    done
    x=$((x + 1))
done

