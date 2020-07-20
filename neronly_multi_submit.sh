/usr/local/bin/nosh
output_dir=$1
epoch_num=10
total_train_steps=20000
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N ner_only_multi
cd ~
pwd


ner_data_folder="biobert_data/datasets/NER"

# Fix the dataset for now
target_train_file=""
for file in $(ls ${ner_data_folder})
do
    folder_path=$ner_data_folder"/"$file
    if [ -d $folder_path ]
    then
        train_file=$folder_path"/ent_train_dev.tsv"
        dev_file=$folder_path"/ent_test.tsv"
        echo "Training and testing respectively on "
        echo $train_file
        echo $dev_file
        singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --crf --mode ner --total_train_steps $total_train_steps --output_dir $output_dir  --num_train_epochs $epoch_num --ner_train_files $train_file --ner_dev_file $dev_file --load_model
    fi
done

echo "All results are stored in "$output_dir
