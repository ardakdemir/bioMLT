/usr/local/bin/nosh
output_dir=${1}
load_model_path=${2}
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N qas_only_train
cd ~
if [ -f $load_model_path ]
then
  echo "LOADING PRETRAINED MODEL "${load_model_path}
  singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --output_dir $output_dir  --load_model_path ${load_model_path} --max_seq_length 256 --num_train_epochs 5
else
  echo "NOT LOADING MODEL. STARTING WITH Biobert v1.1"
  singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --output_dir $output_dir  --load_model --max_seq_length 256 --num_train_epochs 5  --batch_size 12 --load_model
fi
echo "All results are stored in "$output_dir
