/usr/local/bin/nosh
output_dir='neronly_1504_3'
#test_num=$2
#squad_train_file='../biobert_data/qas_train_split.json'
#squad_eval_file='../biobert_data/qas_dev_split.json'
epoch_num=10
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N ner_only_train
cd ~/bioMLT
pwd


ner_data_folder="ner_data/bioner_data"
for file in $(ls ${ner_data_folder})
do
    folder_path=$ner_data_folder"/"$file
    if [ -d $folder_path ]
    then
        train_file=$folder_path"/bio_"$file"_train_dev.tsv"
        dev_file=$folder_path"/bio_"$file"_test.tsv"
        echo "Training and testing respectively on "
        echo $train_file
        echo $dev_file
        singularity exec --nv ~/singularity/pt-cuda-tf python biomlt_alldata.py --mode ner --output_dir $output_dir  --num_train_epochs $epoch_num --ner_train_file $train_file --ner_dev_file $dev_file
    fi
done

echo "All results are stored in "$output_dir
