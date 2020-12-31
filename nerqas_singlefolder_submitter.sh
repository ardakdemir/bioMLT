#!/bin/bash

output_folder=$1
ner_folder=$2
pref_name=$3

root_folder="/home/aakdemir/"
# Fix the dataset for now


echo "Dataset folder: "$ner_folder
datasets=$(ls ${ner_folder})

echo "NER Output dir: "$output_dir
for target_fd in $datasets
do
  echo "Dataset path "$subset_folder_path
  echo "Target dataset: "$target_fd
  experiment_name="nerqas_"${pref_name}"_"$target_fd
  qsub -N $experiment_name /home/aakdemir/bioMLT/nerqas_sequential_submit.sh $output_folder $ner_folder $target_fd
done

