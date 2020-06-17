/usr/local/bin/nosh
output_dir='joint_1706'
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N joint_flat_train
cd ~
singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --mode joint_flat --output_dir $output_dir  --load_model --max_seq_length 256 --num_train_epochs 50

echo "All results are stored in "$output_dir
