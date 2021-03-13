#!/bin/bash

output_dir=$1
pref_name=$2
ner_data_folder=$3
if [ -d $output_dir ]
then
  rm $output_dir
fi

datasets=$(ls ${ner_data_folder})
# Fix the dataset for now
echo $datasets
target=0
for target_fd in $datasets
do
    echo $target_fd
    folder_path=$ner_data_folder"/"$target_fd
    qsub -N $pref_name"_target_"$target_fd /home/aakdemir/bioMLT/neronly_initbert_stl_submit.sh $output_dir $ner_data_folder $target_fd
done

