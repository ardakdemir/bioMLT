/usr/local/bin/nosh
output_dir=$1
epoch_num=10
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N ner_only_train_crf_mtldatasets
cd ~
pwd

dataset_list=("BC5CDR-chem" "BC5CDR-disease" "NCBI-disease" "JNLPBA" "BC4CHEMD" "linnaeus")

ner_data_folder="MTL-Bioinformatics-2016/data"
for file in $(ls ${ner_data_folder})
do
    echo $file
    folder_path=$ner_data_folder"/"$file
    if [[ $folder == *"-IOB" ]]
    then
        x=$(echo $folder | rev | cut -c 5- | rev)
        if [[ " ${init_list[@]} " =~ " ${x} " ]]
        then
    		echo "Training for "$file
            train_file=$folder_path"/train_dev.tsv"
            dev_file=$folder_path"/test.tsv"
            echo "Training and testing respectively on "
            echo $train_file
            echo $dev_file
            singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --crf --mode ner --output_dir $output_dir  --num_train_epochs $epoch_num --ner_train_file $train_file --ner_dev_file $dev_file --load_model
        fi
    fi
done

echo "All results are stored in "$output_dir
