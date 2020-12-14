/usr/local/bin/nosh
output_dir=${1}
ner_folder=${2}
ner_dataset_name=${3}
epoch_num=3
ner_latent_dim=64
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N qas_hier_train

load_model_path=${ner_folder}"/best_ner_model_on_"${ner_dataset_name}
load_ner_vocab_path=${load_model_path}"_vocab"

singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --ner_latent_dim $ner_latent_dim --ner_dataset_name ${ner_dataset_name} --mode qas --output_dir $output_dir  --load_model_path ${load_model_path} --load_ner_label_vocab_path ${load_ner_vocab_path} --max_seq_length 512 --num_train_epochs $epoch_num  --qas_with_ner --init_ner_head --crf --model_save_name best_qas_model_${ner_dataset_name}
