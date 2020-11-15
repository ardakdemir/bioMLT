/usr/local/bin/nosh
output_dir=${1}
load_model_path=${2}
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N qas_only_train
singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --output_dir $output_dir  --load_model_path ${load_model_path} --max_seq_length 256 --num_train_epochs 50
