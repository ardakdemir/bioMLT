/usr/local/bin/nosh
output_dir='save_dir'
model_path=$1
test_path=$2
sys_name=$3
save_folder=$4
#test_num=$2
#squad_train_file='../biobert_data/qas_train_split.json'
#squad_eval_file='../biobert_data/qas_dev_split.json'
epoch_num=5
bioasq_dataset_folder='/home/aakdemir/biobert_data/datasets/QA/BioASQ/'
train_root='/home/aakdemir/biobert_data/BioASQ-training9b/'
bioasq_9bfolder='/home/aakdemir/biobert_data/BioASQ-9b/'

nbest_path='nbest_pred_'${sys_name}
pred_path='pred_'${sys_name}
#EVAL_PATH='/home/aakdemir/biobert_data/Evaluation-Measures'

my_n2b_yesno_path='mytransform_n2b_yesno.py'
my_n2b_list_path='mytransform_n2b_list.py'
my_n2b_factoid_path='mytransform_n2b_factoid.py'
#result_file='qas_result_'${model_save_name}
#init_result_file=$result_file
appended_folder='Appended-Snippet'
fullabs_folder='Full-Abstract'
snippet_folder='Snippet-as-is'
# squad_predict_file=${bioasq_dataset_folder}'BioASQ-test-factoid-6b-'${test_num}'.json'
#pretrained_biobert_model_path='../biobert_data/biobert-squadv1.1_pretrained'
pretrained_biobert_model_path='../biobert_data/biobert_v1.1_pubmed'
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N bioasq_8b_predict

echo $BIOBERT_PATH
echo $n2bfactoid_path
echo $out_for_bioasq_eval
echo "Fine tuning : "${finetuned_model}
echo "Model to be saved:  "$model_save_name
echo $squad_predict_file
cd ~/bioMLT

basename=$(basename ${test_path})
dirname=$(dirname ${test_path})

echo "Basename and Dirname"
echo $basename
echo $dirname

load_model_path=${model_path}
#rm $nbest_path
echo "Loading model from "$load_model_path
test_suff='_squadformat_test_'
out_for_bioasq_eval='converted_'${sys_name}'_'${basename}
out_json='merged_json_'${sys_name}'_'${basename}

python bioasq_squad_converter.py ${test_path} 'test'
squad_predict_yesno_file=${dirname}'/'${basename}${test_suff}'yesno.json'
squad_predict_list_file=${dirname}'/'${basename}${test_suff}'list.json'
squad_predict_factoid_file=${dirname}'/'${basename}${test_suff}'factoid.json'
echo $squad_predict_list_file



singularity exec --nv ~/singularity/pt-cuda-tf python biomlt_alldata.py --predict --load_model_path $load_model_path  --nbest_path $nbest_path --squad_predict_factoid_file $squad_predict_factoid_file --squad_predict_list_file $squad_predict_list_file --squad_predict_yesno_file $squad_predict_yesno_file  --pred_path $pred_path
#rm $out_for_bioasq_eval

python ${my_n2b_yesno_path}  --nbest_path $pred_path'_yesno.json' --output_path $out_for_bioasq_eval'_yesno'
python ${my_n2b_factoid_path} --nbest_path $nbest_path'_factoid.json' --output_path $out_for_bioasq_eval'_factoid'
python ${my_n2b_list_path} --nbest_path $nbest_path'_list.json' --output_path $out_for_bioasq_eval'_list'
echo "Storing all intermediate files with prefix "${out_for_bioasq_eval}

python merge_predictions.py ${out_for_bioasq_eval}  ${out_json}

python prepare_submit_bioasq.py $sys_name ${out_json} ${test_path}
