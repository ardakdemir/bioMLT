/usr/local/bin/nosh
output_dir=${1}
ner_dataset_folder=${2}
ner_dataset_name=${3}
ner_latent_dim=64
ner_epoch_num=10
total_train_steps=-1
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G

folder_path=$ner_dataset_folder"/"$ner_dataset_name
target_train_file=$folder_path"/ent_train.tsv"
target_test_file=$folder_path"/ent_test.tsv"
target_dev_file=$folder_path"/ent_devel.tsv"



#First NER
singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --init_bert --qas_with_ner --crf --mode ner --total_train_steps $total_train_steps --output_dir $output_dir  --num_train_epochs $epoch_num --ner_train_file  $target_train_file  --ner_dev_file $target_dev_file  --ner_test_file $target_test_file  --load_model


load_model_path=${output_dir}"/best_ner_model_on_"${ner_dataset_name}
load_ner_vocab_path=${load_model_path}"_vocab"

#Second QAS
singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --init_bert --ner_latent_dim --mode qas $ner_latent_dim --ner_dataset_name ${ner_dataset_name} --mode qas --output_dir $output_dir  --load_model_path ${load_model_path} --load_ner_label_vocab_path ${load_ner_vocab_path} --max_seq_length 256  --qas_with_ner --init_ner_head --crf --model_save_name best_qas_model_${ner_dataset_name}
