/usr/local/bin/nosh
output_dir=${1}
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N qas_only_train_init
cd ~

echo "NOT LOADING MODEL. STARTING WITH BERT"
singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --init_bert --output_dir $output_dir  --load_model --max_seq_length 256 --num_train_epochs 3  --batch_size 12
echo "All results are stored in "$output_dir
