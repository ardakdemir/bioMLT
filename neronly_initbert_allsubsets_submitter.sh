#!/bin/bash

output_pref=$1
ner_folder_pref=$2
pref_name=$3

pref_length=$(expr length $ner_folder_pref)
ner_data_folder_pref=$ner_folder_pref
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
    exp_code=${subset_folder:$pref_length}
    output_dir=$output_pref"_"$exp_code
    y=0
    echo "NER Output dir: "$output_dir
    echo "QAS Output dir: "qas_output_dir
    for target_fd in $datasets
    do
      folder_path=$subset_folder_path"/"$target_fd
      echo "Dataset path "$folder_path
      echo "Target dataset: "$target_fd
      y=$((y + 1))
      ner_experiment_name="NER_"${pref_name}"_"${exp_code}
      qas_experiment_name="QAS_"${pref_name}"_"${exp_code}
      echo "NER Experiment name: "$ner_experiment_name
      qsub -N $ner_experiment_name /home/aakdemir/bioMLT/neronly_initbert_stl_submit.sh $output_dir $subset_folder_path $target_fd
      echo "QAS Experiment name: "$qas_experiment_name
      qsub -N $qas_experiment_name /home/aakdemir/bioMLT/qashier_submit.sh $output_dir $output_dir $target_fd
    done
    x=$((x + 1))
done

