/usr/local/bin/nosh
output_dir=$1
target_dataset=$2
epoch_num=10
total_train_steps=-
#$ -cwd
#$ -l v100=1,s_vmem=100G,mem_req=100G
cd ~
ner_data_folder="biobert_data/datasets/NER_for_QAS"
datasets=$(ls ${ner_data_folder})
# Fix the dataset for now

folder_path=$ner_data_folder"/"$target_dataset
target_train_file=$folder_path"/ent_train.tsv"
target_test_file=$folder_path"/ent_test.tsv"
target_dev_file=$folder_path"/ent_devel.tsv"

singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --qas_with_ner --crf --mode ner --total_train_steps $total_train_steps --output_dir $output_dir  --num_train_epochs $epoch_num --ner_train_file  $target_train_file  --ner_dev_file $target_dev_file  --ner_test_file $target_test_file  --load_model
echo "Results are stored in "$output_dir