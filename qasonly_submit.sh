/usr/local/bin/nosh
output_dir=${1}

#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N qas_only_train
cd ~
singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --mode qas --model_save_name "best_qas_model_woner" --output_dir $output_dir  --load_model
echo "All results are stored in "$output_dir
