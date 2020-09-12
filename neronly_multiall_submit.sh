/usr/local/bin/nosh
output_dir=$1
if [ -d $output_dir ]
then
  rm $output_dir
fi
epoch_num=10
total_train_steps=320000
#$ -cwd
#$ -l v100=1,s_vmem=100G,mem_req=100G
#$ -N ner_only_multi
cd ~
ner_data_folder="biobert_data/datasets/NER"
datasets=$(ls ${ner_data_folder})
# Fix the dataset for now
singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --crf --all --mode multiner --total_train_steps $total_train_steps --output_dir $output_dir  --num_train_epochs $epoch_num  --load_model


echo "All results are stored in "$output_dir
