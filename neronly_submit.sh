/usr/local/bin/nosh
output_dir='neronly_1606'
epoch_num=20
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N ner_only_train
cd ~
pwd


ner_data_folder="biobert_data/datasets/NER/"
for file in $(ls ${ner_data_folder})
do
    folder_path=$ner_data_folder"/"$file
    if [ -d $folder_path ]
    then
        train_file=$folder_path"/ENT_"$file"_train_dev.tsv"
        dev_file=$folder_path"/ENT_"$file"_test.tsv"
        echo "Training and testing respectively on "
        echo $train_file
        echo $dev_file
        singularity exec --nv ~/singularity/pt-cuda-tf python biomlt_alldata.py --mode ner --output_dir $output_dir  --num_train_epochs $epoch_num --ner_train_file $train_file --ner_dev_file $dev_file --load_model
    fi
done

echo "All results are stored in "$output_dir
echo "All results are stored in "$output_dir
