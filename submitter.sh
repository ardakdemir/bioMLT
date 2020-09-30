#!/bin/bash

output_dir=$1
if [ -d $output_dir ]
then
  rm $output_dir
fi

pref_name="cli_multitasks_0110"

ner_data_folder="biobert_data/datasets/NER"
datasets=$(ls ${ner_data_folder})
# Fix the dataset for now
target=0
for target_fd in $datasets
do
    folder_path=$ner_data_folder"/"$target_fd
    if [ -d $folder_path ]
    echo "Iteration for "$folder_path
    then
        aux=0
        for aux_fd in $datasets
        do
            aux_folder_path=$ner_data_folder"/"$aux_fd
            if [ -d $aux_folder_path ]
            then
                if [ $target -le $aux ]
                then
                  qsub -N $pref_name"_target_"$target_fd"_aux_"$aux_fd neronly_mtl_single_submit.sh $output_dir $target_fd $aux_fd
               fi
            fi

            aux=$(($aux + 1))
        done
    fi
    target=$(($target + 1))
done

