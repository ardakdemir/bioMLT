#!/bin/bash

output_pref=$1
ner_folder=$2
pref_name=$3
if [ -d $output_dir ]
then
  rm $output_dir
fi

ner_data_folder_pref=$ner_folder
ner_data_folder_pref=$ner_data_folder_pref*
root_folder="/home/aakdemir/"
dataset_folders=$(ls -d ${ner_data_folder_pref})
# Fix the dataset for now
x=0
for subset_folder in $dataset_folders
do
    echo "Subset folder: "$subset_folder
    subset_folder_path=${subset_folder}
    datasets=$(ls ${subset_folder_path})
    exp_code=${subset_folder:55}
    output_dir=$output_pref"_"$exp_code
    y=0
    echo "Output dir: "$output_dir
    for target_fd in $datasets
    do
      folder_path=$subset_folder_path"/"$target_fd
      echo "Dataset path "$folder_path
      y=$((y + 1))
      experiment_name=${pref_name}"_"${exp_code}
      echo "Experiment name: "$experiment_name
      qsub -N $experiment_name /home/aakdemir/bioMLT/neronly_stl_submit.sh $output_dir $subset_folder_path $target_fd
    done
    x=$((x + 1))
done

