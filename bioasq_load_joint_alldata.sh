/usr/local/bin/nosh
output_dir='save_dir'
finetuned_model=$1
model_save_name=$2
predict=$3
#test_num=$2
#squad_train_file='../biobert_data/qas_train_split.json'
#squad_eval_file='../biobert_data/qas_dev_split.json'
epoch_num=10
bioasq_dataset_folder='/home/aakdemir/biobert_data/datasets/QA/BioASQ/'
bioasq_preprocessed_folder='/home/aakdemir/biobert_data/BioASQ-6b/'

nbest_path='nbest_pred_'${model_save_name}
pred_path='pred_'${model_save_name}
EVAL_PATH='/home/aakdemir/biobert_data/Evaluation-Measures'

BIOBERT_PATH='/home/aakdemir/bioasq-biobert/'
n2bfactoid_path='biocodes/transform_n2b_factoid.py'
n2byesno_path='./biocodes/transform_n2b_yesno.py'
myn2byesno_path='mytransform_n2b_yesno.py'
my_n2b_list_path='mytransform_n2b_list.py'
result_file='qas_result_'${model_save_name}
init_result_file=$result_file

# squad_predict_file=${bioasq_dataset_folder}'BioASQ-test-factoid-6b-'${test_num}'.json'
#pretrained_biobert_model_path='../biobert_data/biobert-squadv1.1_pretrained'
pretrained_biobert_model_path='../biobert_data/biobert_v1.1_pubmed'
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N all_data_train

echo $EVAL_PATH
echo $BIOBERT_PATH
echo $n2bfactoid_path
echo $out_for_bioasq_eval
echo "Fine tuning : "${finetuned_model}
echo "Model to be saved:  "$model_save_name
echo $squad_predict_file
cd ~/bioMLT

pwd

if [ $predict = 0 ]
then
    echo " PREDICT IS "${predict}
    # First we train a model on the bioasq dataset and then test it on the test set!! 
    # Used for comparing the results for different pretrained models
    echo "Running training on all question types !! "
    echo "First experimenting with qas aloneee"
    echo "Best model will be saved in "${output_dir}/${model_save_name}
    singularity exec --nv ~/singularity/pt-cuda-tf python biomlt_alldata.py --biobert_model_path $pretrained_biobert_model_path --model_save_name $model_save_name --output_dir $output_dir  --num_train_epochs $epoch_num  --overwrite_cache  --load_model_path $finetuned_model 
    load_model_path=${output_dir}"/"${model_save_name}
else
    load_model_path=${finetuned_model}
    echo "SKIPPING TRAINING  - MOVING TO PREDICTIONS"
fi
#rm $nbest_path
echo "Loading model from "$load_model_path

