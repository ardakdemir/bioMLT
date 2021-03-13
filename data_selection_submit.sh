/usr/local/bin/nosh
output_dir=${1}
load_model_path=${2}
load_ner_label_vocab_path=${3}
#QAS Model is given input of the embeddings

#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N qas_only_train
singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --mode qas --crf --output_dir $output_dir  --load_model_path $load_model_path --max_seq_length 256 --init_ner_head  --qas_with_ner --num_train_epochs 50 --load_ner_label_vocab_path $load_ner_label_vocab_path

echo "All results are stored in "$output_dir
