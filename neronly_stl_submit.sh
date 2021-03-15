/usr/local/bin/nosh
output_dir=$1
ner_data_folder=$2
target_dataset=$3
epoch_num=10
total_train_steps=-1
#$ -cwd
#$ -l v100=1,s_vmem=100G,mem_req=100G
cd ~

# Fix the dataset for now

folder_path=$ner_data_folder"/"$target_dataset
target_train_file=$folder_path"/ent_train.tsv"
target_test_file=$folder_path"/ent_test.tsv"
target_dev_file=$folder_path"/ent_devel.tsv"

singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py  --crf --mode ner --total_train_steps $total_train_steps --output_dir $output_dir  --num_train_epochs $epoch_num --ner_train_file  $target_train_file  --ner_dev_file $target_dev_file  --ner_test_file $target_test_file  --load_model
echo "Results are stored in "$output_dir
