/usr/local/bin/nosh
output_dir='save_dir'
model_save_name=$1

#test_num=$2
squad_train_file='train-v1.1.json'
squad_predict_file='dev-v1.1.json'
epoch_num=5
bioasq_dataset_folder='/home/aakdemir/biobert_data/datasets/QA/BioASQ/'
bioasq_preprocessed_folder='/home/aakdemir/biobert_data/BioASQ-6b/'

nbest_path='biomlt_train_pred_nbest_pred'
EVAL_PATH='/home/aakdemir/biobert_data/Evaluation-Measures'

BIOBERT_PATH='/home/aakdemir/bioasq-biobert'
n2bfactoid_path='./biocodes/transform_n2b_factoid.py'
n2byesno_path='./biocodes/transform_n2b_factoid.py'
out_for_bioasq_eval='input_for_bioasq_'${model_save_name}
# squad_predict_file=${bioasq_dataset_folder}'BioASQ-test-factoid-6b-'${test_num}'.json'
#pretrained_biobert_model_path='../biobert_data/biobert-squadv1.1_pretrained'
pretrained_biobert_model_path='../biobert_data/biobert_v1.1_pubmed'
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N finetune_pubmed 

echo $EVAL_PATH
echo $BIOBERT_PATH
echo $n2bfactoid_path
echo $out_for_bioasq_eval
echo $gold_path
echo "Fine tuning : "$pretrained_biobert_model_path
echo "Model to be saved:  "$model_save_name
echo $squad_predict_file
cd ~/bioMLT

pwd
# First we train a model on the bioasq dataset and then test it on the test set!! 
# Used for comparing the results for different pretrained models
singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py  --squad_dir squad_data  --biobert_model_path $pretrained_biobert_model_path --model_save_name $model_save_name --output_dir $output_dir  --num_train_epochs $epoch_num  --overwrite_cache --squad_train_file $squad_train_file --squad_predict_file $squad_predict_file --only_squad

