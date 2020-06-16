/usr/local/bin/nosh
output_dir='qasonly_1606'
epoch_num=30
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N qas_only_train
cd ~
singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --output_dir qas_experiments  --load_model --max_seq_length 256 --num_train_epochs 50

echo "All results are stored in "$output_dir
