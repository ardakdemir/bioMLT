/usr/local/bin/nosh
train_file=$1
test_file=$2
output_dir=$3
#test_num=$2
#squad_train_file='../biobert_data/qas_train_split.json'
#squad_eval_file='../biobert_data/qas_dev_split.json'
epoch_num=30
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N ner_o
cd ~/bioMLT
pwd


ner_data_folder="ner_data/bioner_data"
singularity exec --nv ~/singularity/pt-cuda-tf python biomlt_alldata.py --mode ner --output_dir $output_dir  --num_train_epochs $epoch_num --ner_train_file $train_file --ner_dev_file $test_file

echo "All results are stored in "$output_dir
