#!/bin/bash

output_dir_pref=$1
ner_datasets_pref=$2
ner_folders_pref=$3
exp_name=$4

l=$(expr length "${ner_datasets_pref}")
echo $l

ner_dataset_folders=$(ls -d ${ner_datasets_pref}*)
ner_folders=$(ls -d ${ner_folders_pref}*)
# Fix the dataset for now
for folder in $ner_folders
do
  exp_code=${folder:$l}
  ner_folder=$ner_datasets_pref"_"$exp_code
  echo "NER dataset folder: "$ner_folder
  output_dir=$output_dir_pref"_"$exp_code
  datasets=$(ls ${ner_folder})
  echo "All datasets: "$datasets
  if [ -d $output_dir ]
  then
    rm $output_dir
  fi
  for target_fd in $datasets
  do
      echo "Submitting "${target_fd}
      echo "Ner  dir: "$ner_folder
      echo "Output dir: "$output_dir
      echo "Target dataset: "$target_fd
      #qsub -N $pref_name"_nermodel_"$target_fd /home/aakdemir/bioMLT/qashier_submit.sh $output_dir $ner_folder $target_fd
  done
done

