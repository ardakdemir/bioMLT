/usr/local/bin/nosh
output_dir=${1}
ner_model_path=${2}
ner_vocab_path=${3}
ner_name=${4}


#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
cd ~
singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py   --init_ner_head --mode qas --ner_dataset_name ${ner_name} --output_dir ${output_dir}   --max_seq_length 256 --model_save_name best_qas_model_${ner_name} --load_model_path ${ner_model_path} --load_ner_label_vocab_path ${ner_vocab_path}
echo "All results are stored in "$output_dir
