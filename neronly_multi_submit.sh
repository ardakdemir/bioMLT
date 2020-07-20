/usr/local/bin/nosh
output_dir=$1
epoch_num=5
total_train_steps=2000
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N ner_only_multi
cd ~
ner_data_folder="biobert_data/datasets/NER"
datasets=$(ls ${ner_data_folder})
# Fix the dataset for now
target_train_file=""
target=0
for target_fd in $datasets
do
    folder_path=$ner_data_folder"/"target_fd
    if [ -d $folder_path ]
    echo "Iteration for "$folder_path
    then
        aux=0
        for aux_fd in $datasets
        do
            aux_folder_path=$ner_data_folder"/"aux_file
            if [ -d $aux_folder_path ]
            then
                if [ $target -le $aux ]
                then
                  target_train_file=$folder_path"/ent_train_dev.tsv"
                  target_dev_file=$folder_path"/ent_test.tsv"
                  aux_train_file=$aux_folder_path"/ent_train_dev.tsv"
                  aux_dev_file=$aux_folder_path"/ent_test.tsv"
                  echo "Training and testing respectively on "
                  echo $target_train_file  $aux_train_file
                  echo $target_dev_file $aux_dev_file
                  singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --crf --mode multiner --total_train_steps $total_train_steps --output_dir $output_dir  --num_train_epochs $epoch_num --ner_train_files  $target_train_file  $aux_train_file --ner_dev_file $target_dev_file $aux_dev_file --load_model
               fi
            fi

            aux=$(($aux + 1))
        done
    fi
    target=$(($target + 1))
done

echo "All results are stored in "$output_dir
